use crate::{
    routes::openai::{ChatCompletionMessageContentPart, ChatCompletionMessageParams},
    tokenizer::TokenizerConfig,
};
use anyhow::anyhow;
use minijinja::Environment;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const DEFAULT_TEMPLATE: &str = r#"{{- bos_token }}
{%- for message in messages %}
    {{- '<|' + message['role'] + |>\n' }}
    {{- message['content'] + eos_token }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- endif %}"#;

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
    tools: Option<Vec<Value>>,
    documents: Option<Vec<ChatDocument>>,
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

impl ChatBuilder {
    pub fn new(config: &TokenizerConfig) -> anyhow::Result<Self> {
        let default_context = TemplateContext {
            messages: vec![],
            tools: None,
            documents: None,
            add_generation_prompt: true,
            tools_in_user_message: None,
            date_string: date_string(),
            bos_token: config.bos_token.clone(),
            eos_token: config.eos_token.clone(),
            unk_token: config.unk_token.clone(),
            sep_token: config.sep_token.clone(),
            pad_token: config.pad_token.clone(),
            cls_token: config.cls_token.clone(),
            mask_token: config.mask_token.clone(),
        };
        let mut env = Environment::new();
        // https://github.com/huggingface/transformers/blob/e50bf61decf741c6d59e4ba633b7392712673bda/src/transformers/utils/chat_template_utils.py#L423
        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);
        let template = config
            .chat_template
            .clone()
            .unwrap_or_else(|| DEFAULT_TEMPLATE.to_string());
        log::info!("chat template:\n{}", template);
        env.add_template_owned("chat", template)
            .map_err(|e| anyhow!("error parsing chat_template: {}", e))?;
        let res = Self {
            default_context,
            env,
        };
        // make sure the template is valid
        let msg = res.build(&vec![
            ChatCompletionMessageParams::System {
                content: Some("Be a good model".to_string()),
                name: None,
            },
            ChatCompletionMessageParams::User {
                content: ChatCompletionMessageContentPart::Text("Hello".to_string()),
                name: None,
            },
        ])?;
        log::info!("chat template result:\n{}", msg);
        Ok(res)
    }

    pub fn build(&self, messages: &Vec<ChatCompletionMessageParams>) -> anyhow::Result<String> {
        let mut context = self.default_context.clone();
        context.messages = messages.iter().map(|x| x.flatten()).collect();
        context.date_string = date_string();
        let context_non_null = serde_json::to_value(&context)?
            .as_object()
            .unwrap()
            .iter()
            .fold(serde_json::Map::new(), |mut acc, (k, v)| {
                if !v.is_null() {
                    acc.insert(k.clone(), v.clone());
                }
                acc
            });
        let r = self.env.get_template("chat")?.render(&context_non_null)?;
        Ok(r)
    }
}

impl ChatCompletionMessageParams {
    // HF templates generally don't support multiple content parts, so we flatten them here
    pub fn flatten(&self) -> Self {
        match self {
            ChatCompletionMessageParams::User {
                content: ChatCompletionMessageContentPart::ContentParts(parts),
                name,
            } => ChatCompletionMessageParams::User {
                content: ChatCompletionMessageContentPart::Text(
                    parts
                        .iter()
                        .map(|part| part.text.clone())
                        .collect::<Vec<_>>()
                        .join("\n"),
                ),
                name: name.clone(),
            },
            x => x.clone(),
        }
    }
}
