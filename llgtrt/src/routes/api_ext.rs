use llguidance::api::TopLevelGrammar;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::openai::{
    ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParams, Completion, CompletionChoice,
    FinishReason, LogProbs, Tool,
};

#[derive(Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum LlgLogLevel {
    None,
    Warning,
    Json,
    Verbose,
}

impl Default for LlgLogLevel {
    fn default() -> Self {
        LlgLogLevel::None
    }
}

impl LlgLogLevel {
    pub fn has_json(&self) -> bool {
        matches!(self, LlgLogLevel::Json | LlgLogLevel::Verbose)
    }
    pub fn to_log_level(&self) -> u32 {
        match self {
            LlgLogLevel::None => 0,
            LlgLogLevel::Warning => 1,
            LlgLogLevel::Json => 1,
            LlgLogLevel::Verbose => 2,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RunControllerArg {
    pub grammar: TopLevelGrammar,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RunRequest {
    pub controller: String,
    pub controller_arg: RunControllerArg,
    pub prompt: Option<String>,
    pub messages: Option<Vec<ChatCompletionMessageParams>>,
    pub temperature: Option<f32>,  // defl 0.0
    pub top_p: Option<f32>,        // defl 1.0
    pub max_tokens: Option<usize>, // defl context size
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunUsageResponse {
    pub sampled_tokens: usize,
    pub ff_tokens: usize,
    pub cost: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitialRunResponse {
    pub id: String,
    pub object: &'static str, // "initial-run"
    pub created: u64,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResponse {
    pub object: &'static str, // "run"
    pub forks: Vec<RunForkResponse>,
    pub usage: RunUsageResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunForkResponse {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    pub text: String,
    pub error: String,
    pub logs: String,
    pub storage: Vec<serde_json::Value>,
    pub micros: u64,
    pub logprobs: Option<LogProbs>,
}

impl Completion {
    pub fn of_chat_completion(chat_completion: ChatCompletion) -> Completion {
        let choices = chat_completion
            .choices
            .into_iter()
            .map(|choice| CompletionChoice {
                text: choice.message.content.unwrap_or_default(),
                index: choice.index,
                logprobs: None,
                finish_reason: choice.finish_reason,
                llg_logs: choice.llg_logs,
            })
            .collect();

        Completion {
            id: chat_completion.id,
            object: "text_completion".to_string(),
            created: chat_completion.created,
            model: chat_completion.model,
            choices,
            usage: chat_completion.usage,
        }
    }

    pub fn of_chat_completion_chunk(chat_completion_chunk: ChatCompletionChunk) -> Completion {
        let choices = chat_completion_chunk
            .choices
            .into_iter()
            .map(|choice| CompletionChoice {
                text: choice.delta.content.unwrap_or_default(),
                index: choice.index,
                logprobs: None,
                finish_reason: choice.finish_reason,
                llg_logs: choice.llg_logs,
            })
            .collect();

        Completion {
            id: chat_completion_chunk.id,
            object: "text_completion".to_string(),
            created: chat_completion_chunk.created,
            model: chat_completion_chunk.model,
            choices,
            usage: chat_completion_chunk.usage,
        }
    }
}

impl Tool {
    pub fn to_schema(&self) -> Value {
        match self {
            Tool::Function { function } => {
                let params = if function.strict == Some(true) {
                    function.parameters.clone()
                } else {
                    json!({
                        "type": "object"
                    })
                };
                json!({
                    "type": "object",
                    "required": ["type", "name", "parameters"],
                    "additionalProperties": false,
                    "properties": {
                        "type": { "const": "function" },
                        "name": { "const": function.name.clone() },
                        "parameters": params,
                    }
                })
            }
        }
    }
}

pub fn tools_to_schema(tools: &Vec<Tool>) -> Value {
    if tools.is_empty() {
        // ???
        json!({
            "type": "object"
        })
    } else if tools.len() == 1 {
        tools[0].to_schema()
    } else {
        json!({
            "anyOf": tools.iter().map(Tool::to_schema).collect::<Vec<_>>(),
        })
    }
}
