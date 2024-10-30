use super::api_ext::LlgLogLevel;
use llguidance_parser::api::TopLevelGrammar;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use toktrie::{TokTrie, TokenId};

// https://platform.openai.com/docs/api-reference/chat/create
#[derive(Deserialize, Debug)]
pub struct ChatCompletionCreateParams {
    /// A list of messages comprising the conversation so far.
    pub messages: Vec<ChatCompletionMessageParams>,

    #[serde(default)]
    pub logprobs: bool,

    pub top_logprobs: Option<usize>,

    /// A list of tools the model may call.
    #[serde(default)]
    pub tools: Vec<Tool>,

    /// Controls which (if any) tool is called by the model.
    #[serde(default)]
    pub tool_choice: ToolChoice,

    /// Whether to include the JSON schema of tools or response format when using chat template.
    /// Defaults to true.
    #[serde(default)]
    pub include_json_schema_in_prompt: Option<bool>,

    #[serde(flatten)]
    pub params: CommonCreateParams,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(untagged)]
pub enum ToolChoice {
    Simple(ToolChoiceOption),
    Advanced(ToolChoiceAdvanced),
}

impl Default for ToolChoice {
    fn default() -> Self {
        ToolChoice::Simple(ToolChoiceOption::Auto)
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoiceAdvanced {
    Function { name: String },
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceOption {
    /// The model can pick between generating a message or calling one or more tools.
    Auto,
    /// The model will not call any tool and instead generates a message
    None,
    /// The model must call one or more tools.
    Required,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Tool {
    Function { function: FunctionTool },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionTool {
    /// The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
    pub name: String,
    /// A description of what the function does, used by the model to choose when and how to call the function.
    pub description: Option<String>,
    /// A JSON schema for the 'parameters' object.
    #[serde(default)]
    pub parameters: serde_json::Value,
    /// Whether to enable strict schema adherence when generating the function call. If set to true, the model will follow the exact schema defined in the parameters field. Only a subset of JSON Schema is supported when strict is true.
    pub strict: Option<bool>,
}

// https://platform.openai.com/docs/api-reference/completions/create
#[derive(Deserialize, Debug)]
pub struct CompletionCreateParams {
    /// The prompt(s) to generate completions for, encoded as a string, array of strings, array of
    /// tokens, or array of token arrays.
    #[serde(deserialize_with = "string_or_vec")]
    pub prompt: Vec<String>,

    /// The suffix that comes after a completion of inserted text.
    pub suffix: Option<String>,

    /// Generates best_of completions server-side and returns the "best" (the one with the highest
    /// log probability per token). Results cannot be streamed.
    #[serde(default = "default_best_of")]
    pub best_of: usize,

    /// Echo back the prompt in addition to the completion
    #[serde(default)]
    pub echo: bool,

    /// Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens.
    pub logprobs: Option<usize>,

    #[serde(flatten)]
    pub params: CommonCreateParams,
}

#[derive(Deserialize, Debug, Clone)]
pub struct CommonCreateParams {
    /// ID of the model to use.
    pub model: String,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
    /// frequency in the text so far, decreasing the model's likelihood to repeat the same line
    /// verbatim.
    #[serde(default)]
    pub frequency_penalty: f32,
    /// Modify the likelihood of specified tokens appearing in the completion.
    pub logit_bias: Option<HashMap<String, f32>>,
    /// The maximum number of tokens to generate in the completion.
    pub max_tokens: Option<usize>,
    /// An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.
    pub max_completion_tokens: Option<usize>,
    /// How many completions to generate for each prompt.
    #[serde(default = "default_n")]
    pub n: usize,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they
    /// appear in the text so far, increasing the model's likelihood to talk about new topics.
    #[serde(default)]
    pub presence_penalty: f32,
    /// If specified, our system will make a best effort to sample deterministically, such that
    /// repeated requests with the same seed and parameters should return the same result.
    pub seed: Option<u64>,
    /// Up to 4 sequences where the API will stop generating further tokens. The returned text will
    /// not contain the stop sequence.
    pub stop: Option<Vec<String>>,
    /// Whether to stream back partial progress.
    #[serde(default)]
    pub stream: bool,
    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the
    /// output more random, while lower values like 0.2 will make it more focused and deterministic.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// An alternative to sampling with temperature, called nucleus sampling, where the model
    /// considers the results of the tokens with top_p probability mass. So 0.1 means only the
    /// tokens comprising the top 10% probability mass are considered.
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect
    /// abuse.
    #[allow(dead_code)]
    pub user: Option<String>,

    /// An object specifying the format that the model must output.
    /// Setting to { "type": "json_object" } enables JSON mode, which guarantees the message the
    /// model generates is valid JSON.
    pub response_format: Option<ResponseFormat>,

    /// The minimum number of tokens to generate in the completion.
    #[serde(default = "default_min_tokens")]
    pub min_tokens: usize,

    #[serde(default)]
    pub llg_log_level: LlgLogLevel,

    /// When set, return the result of applying the chat template to the messages.
    #[serde(default)]
    pub return_expanded_prompt: Option<bool>,

    #[serde(skip)]
    pub logprobs: Option<usize>,
}

#[derive(Serialize, Debug)]
pub struct Completion {
    /// A unique identifier for the completion.
    pub id: String,
    /// The object type, which is always "text_completion"
    pub object: String,
    /// The Unix timestamp (in seconds) of when the completion was created.
    pub created: u64,
    /// The model used for completion.
    pub model: String,
    /// The list of completion choices the model generated for the input prompt.
    pub choices: Vec<CompletionChoice>,
    /// Usage statistics for the completion request.
    pub usage: Usage,
}

#[derive(Serialize, Debug)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<LogProbs>,
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llg_logs: Option<String>,
}

#[derive(Serialize, Debug, Default, Clone)]
pub struct Usage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,
    /// Number of tokens in the generated completion.
    pub completion_tokens: usize,
    /// Total number of tokens used in the request (prompt + completion).
    pub total_tokens: usize,
}

fn default_best_of() -> usize {
    1
}

fn default_n() -> usize {
    1
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

fn default_min_tokens() -> usize {
    1
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(untagged)]
pub enum ChatCompletionMessageContentPart {
    Text(String),
    ContentParts(Vec<ContentPart>),
}

impl Default for ChatCompletionMessageContentPart {
    fn default() -> Self {
        ChatCompletionMessageContentPart::Text(String::new())
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatCompletionMessageParams {
    System {
        content: ChatCompletionMessageContentPart,
        name: Option<String>,
    },
    User {
        content: ChatCompletionMessageContentPart,
        name: Option<String>,
    },
    Assistant {
        content: Option<ChatCompletionMessageContentPart>,
        name: Option<String>,
        tool_calls: Option<Vec<serde_json::Value>>,
    },
    Tool {
        content: Option<ChatCompletionMessageContentPart>,
        tool_call_id: String,
    },
}

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: JsonSchemaOptions },
    Llguidance { grammar: TopLevelGrammar },
}

#[derive(Deserialize, Debug, Clone)]
pub struct JsonSchemaOptions {
    /// The name of the response format. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
    pub name: Option<String>,
    /// A description of what the response format is for, used by the model to determine how to respond in the format.
    #[allow(dead_code)]
    pub description: Option<String>,
    /// The schema for the response format, described as a JSON Schema object.
    pub schema: Option<serde_json::Value>,
    /// Whether to enable strict schema adherence when generating the output. If set to true, the model will always follow the exact schema defined in the schema field. Only a subset of JSON Schema is supported when strict is true.
    #[serde(default)]
    pub strict: bool,
}

#[derive(Serialize, Debug)]
pub struct ChatCompletion {
    /// A unique identifier for the completion.
    pub id: String,
    /// The object type, which is always "chat.completion"
    pub object: String,
    /// The Unix timestamp (in seconds) of when the completion was created.
    pub created: u64,
    /// The model used for completion.
    pub model: String,
    /// This fingerprint represents the backend configuration that the model runs with.
    pub system_fingerprint: Option<String>,
    /// The list of completion choices the model generated for the input prompt.
    pub choices: Vec<ChatCompletionChoice>,
    /// Usage statistics for the completion request.
    pub usage: Usage,
    /// Expanded prompt, if requested with return_expanded_prompt.
    pub expanded_prompt: Option<String>,
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatCompletionMessage,
    pub logprobs: Option<LogProbs>,
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llg_logs: Option<String>,
}

#[derive(Serialize, Debug, Clone)]
pub struct ChatCompletionMessage {
    /// The role of the author of this message.
    pub role: Role,
    /// The contents of the chunk message.
    pub content: Option<String>,
    // Not supported yet:
    // tool_calls
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model hit a natural stop point or a provided stop sequence.
    Stop,
    /// The maximum number of tokens specified in the request was reached.
    Length,
    // Content was omitted due to a flag from our content filters.
    // ContentFilter,
    // The model called a tool
    // ToolCalls,
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionChunk {
    /// A unique identifier for the chat completion. Each chunk has the same ID.
    pub id: String,
    /// The object type, which is always chat.completion.chunk.
    pub object: String,
    /// The Unix timestamp (in seconds) of when the chat completion was created. Each chunk has
    /// the same timestamp.
    pub created: u64,
    /// The model used for completion.
    pub model: String,
    /// This fingerprint represents the backend configuration that the model runs with.
    pub system_fingerprint: Option<String>,
    /// A list of chat completion choices. Can be more than one if n is greater than 1.
    pub choices: Vec<ChatCompletionChunkChoice>,
    /// Usage statistics for the completion request.
    pub usage: Usage,
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionChunkChoice {
    pub index: usize,
    pub delta: ChatCompletionChunkDelta,
    pub finish_reason: Option<FinishReason>,
    pub logprobs: Option<LogProbs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llg_logs: Option<String>,
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionChunkDelta {
    /// The role of the author of this message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    /// The contents of the chunk message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    // Not supported yet:
    // tool_calls
}

#[allow(dead_code)]
#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

fn string_or_vec<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_any(StringOrVecVisitor)
}

struct StringOrVecVisitor;

impl<'de> serde::de::Visitor<'de> for StringOrVecVisitor {
    type Value = Vec<String>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a string or a sequence of strings")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(vec![value.to_string()])
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut vec = Vec::new();
        while let Some(elem) = seq.next_element::<String>()? {
            vec.push(elem);
        }
        Ok(vec)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TokenLogProb {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}

impl TokenLogProb {
    pub fn new(trie: &TokTrie, token_id: TokenId, logprob: f32) -> Self {
        let bytes = trie.decode(&[token_id]);
        TokenLogProb {
            token: String::from_utf8_lossy(&bytes).to_string(),
            logprob,
            bytes,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TopTokenLogProb {
    #[serde(flatten)]
    pub chosen: TokenLogProb,
    pub top_logprobs: Vec<TokenLogProb>,
}

impl TopTokenLogProb {
    pub fn new(trie: &TokTrie, toks: &Vec<(TokenId, f32)>) -> Self {
        let chosen = toks[0];
        let chosen = TokenLogProb::new(trie, chosen.0, chosen.1);
        let top_logprobs = toks
            .iter()
            .map(|(tok, logprob)| TokenLogProb::new(trie, *tok, *logprob))
            .collect();
        TopTokenLogProb {
            chosen,
            top_logprobs,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LogProbs {
    pub content: Vec<TopTokenLogProb>,
}
