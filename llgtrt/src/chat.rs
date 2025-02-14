use crate::{
    jsonutil,
    routes::openai::{
        ChatCompletionMessageContentPart, ChatCompletionMessageParams, ContentPart, Tool,
    },
    tokenizer::TokenizerConfig,
};
use anyhow::{anyhow, Result};
use minijinja::{value::Kwargs, Environment, Error, ErrorKind, Value};
use serde::{Deserialize, Serialize};

pub struct ChatBuilder {
    default_context: TemplateContext,
    env: Environment<'static>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatDocument {
    title: String,
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TemplateContext {
    messages: Vec<ChatCompletionMessageParams>,
    tools: Option<Vec<Tool>>,
    documents: Option<Vec<ChatDocument>>,
    json_schema: Option<serde_json::Value>,
    date_string: String,
    add_generation_prompt: bool,
    tools_in_user_message: Option<bool>,
    bos_token: Option<String>,
    eos_token: String,
    unk_token: Option<String>,
    sep_token: Option<String>,
    pad_token: Option<String>,
    cls_token: Option<String>,
    mask_token: Option<String>,
}

fn date_string() -> String {
    // 3 October 2024
    chrono::Utc::now().format("%e %B %Y").to_string()
}

fn tojson(value: Value, args: Kwargs) -> Result<Value, Error> {
    let indent = match args.get::<usize>("indent") {
        Ok(val) => val,
        Err(_) => 4,
    };
    args.assert_all_used()?;
    let mut out = Vec::<u8>::new();
    let indentation = " ".repeat(indent);
    let formatter = serde_json::ser::PrettyFormatter::with_indent(indentation.as_bytes());
    let mut s = serde_json::Serializer::with_formatter(&mut out, formatter);
    let v = serde::Serialize::serialize(&value, &mut s)
        .map(|_| unsafe { String::from_utf8_unchecked(out) })
        .map_err(|err| {
            Error::new(ErrorKind::InvalidOperation, "cannot serialize to JSON").with_source(err)
        })?;
    Ok(Value::from_safe_string(v))
}

fn strftime_now(format: &str) -> String {
    chrono::Utc::now().format(format).to_string()
}

impl ChatBuilder {
    pub fn new(config: &TokenizerConfig) -> anyhow::Result<Self> {
        let default_context = TemplateContext {
            messages: vec![],
            tools: None,
            json_schema: None,
            documents: None,
            add_generation_prompt: true,
            tools_in_user_message: None,
            date_string: date_string(),
            eos_token: config.eos_token.name(),
            bos_token: config.bos_token.as_ref().map(|x| x.name()),
            unk_token: config.unk_token.as_ref().map(|x| x.name()),
            sep_token: config.sep_token.as_ref().map(|x| x.name()),
            pad_token: config.pad_token.as_ref().map(|x| x.name()),
            cls_token: config.cls_token.as_ref().map(|x| x.name()),
            mask_token: config.mask_token.as_ref().map(|x| x.name()),
        };
        let mut env = Environment::new();
        // https://github.com/huggingface/transformers/blob/e50bf61decf741c6d59e4ba633b7392712673bda/src/transformers/utils/chat_template_utils.py#L423
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);
        env.add_function("raise_exception", |msg: String| {
            let e = minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg);
            Err::<minijinja::Value, _>(e)
        });
        env.add_function("strftime_now", strftime_now);
        env.add_filter("tojson", tojson);
        let template = config
            .chat_template
            .clone()
            .expect("chat_template should be set in TokenizerConfig");
        log::debug!("chat template:\n{}", template);
        env.add_template_owned("chat", template)
            .map_err(|e| anyhow!("error parsing chat_template: {}", e))?;
        let res = Self {
            default_context,
            env,
        };
        // make sure the template is valid
        let msg = res.build(ChatParams {
            tools: &vec![],
            json_schema: None,
            messages: &vec![
                // Some models do not support System
                // ChatCompletionMessageParams::System {
                //     content: ChatCompletionMessageContentPart::Text("Be a good model".to_string()),
                //     name: None,
                // },
                ChatCompletionMessageParams::User {
                    content: ChatCompletionMessageContentPart::Text("Hello".to_string()),
                    name: None,
                },
            ],
        })?;
        log::info!("chat template result:\n{}", msg);
        Ok(res)
    }

    pub fn build(&self, params: ChatParams) -> anyhow::Result<String> {
        let mut context = self.default_context.clone();
        context.messages = params
            .messages
            .iter()
            .map(|x| x.flatten())
            .collect::<Result<Vec<_>>>()?;
        context.date_string = date_string();
        if params.tools.len() > 0 {
            context.tools = Some(params.tools.clone());
        }
        if let Some(json_schema) = params.json_schema {
            context.json_schema = Some(json_schema.clone());
        }
        let mut context = serde_json::to_value(&context)?;
        jsonutil::remove_null(&mut context);
        let r = self
            .env
            .get_template("chat")
            .unwrap()
            .render(&context)
            .map_err(|e| anyhow!("error rendering chat template: {}", e))?;
        log::debug!("chat template result:\n{}", r);
        Ok(r)
    }
}

impl ContentPart {
    pub fn text(&self) -> Option<String> {
        match self {
            ContentPart::Text { text: s } => Some(s.clone()),
            ContentPart::ImageUrl { .. } => None,
        }
    }
}

impl ChatCompletionMessageContentPart {
    pub fn flatten(&self) -> Result<ChatCompletionMessageContentPart> {
        match self {
            ChatCompletionMessageContentPart::Text(s) => {
                Ok(ChatCompletionMessageContentPart::Text(s.clone()))
            }
            ChatCompletionMessageContentPart::ContentParts(parts) => {
                Ok(ChatCompletionMessageContentPart::Text(
                    parts
                        .iter()
                        .map(|x| x.text().ok_or_else(|| anyhow!("unexpected ImageUrl")))
                        .collect::<Result<Vec<_>>>()?
                        .join("\n"),
                ))
            }
        }
    }
}

impl ChatCompletionMessageParams {
    // HF templates generally don't support multiple content parts, so we flatten them here
    pub fn flatten(&self) -> Result<Self> {
        Ok(match self {
            ChatCompletionMessageParams::User { content, name } => {
                ChatCompletionMessageParams::User {
                    content: content.flatten()?,
                    name: name.clone(),
                }
            }
            ChatCompletionMessageParams::System { content, name } => {
                ChatCompletionMessageParams::System {
                    content: content.flatten()?,
                    name: name.clone(),
                }
            }
            ChatCompletionMessageParams::Assistant {
                content,
                name,
                tool_calls,
            } => ChatCompletionMessageParams::Assistant {
                content: content.as_ref().map(|x| x.flatten()).transpose()?,
                name: name.clone(),
                tool_calls: tool_calls.clone(),
            },
            ChatCompletionMessageParams::Tool {
                content,
                tool_call_id,
            } => ChatCompletionMessageParams::Tool {
                content: content.as_ref().map(|x| x.flatten()).transpose()?,
                tool_call_id: tool_call_id.clone(),
            },
        })
    }
}

#[derive(Debug, Serialize)]
pub struct ChatParams<'a> {
    /// A list of messages comprising the conversation so far.
    pub messages: &'a Vec<ChatCompletionMessageParams>,
    pub tools: &'a Vec<Tool>,
    pub json_schema: Option<&'a serde_json::Value>,
}
