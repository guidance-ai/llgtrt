use crate::{
    chat::ChatBuilder,
    config::{CliConfig, LlgTrtConfig},
};
use anyhow::ensure;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use toktrie::{TokEnv, TokEnvWithTrie};

const DEFAULT_TEMPLATE: &str = r#"{{- bos_token }}
{%- for message in messages %}
    {{- '<|' + message['role'] + |>\n' }}
    {{- message['content'] + eos_token }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- endif %}"#;

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub chat_template: Option<String>,

    #[serde(default)]
    pub clean_up_tokenization_spaces: bool,

    pub eos_token: String,
    pub bos_token: Option<String>,
    pub unk_token: Option<String>,
    pub sep_token: Option<String>,
    pub pad_token: Option<String>,
    pub cls_token: Option<String>,
    pub mask_token: Option<String>,

    /// This is <|python_tag|> for Llama 3 models.
    pub json_start_token: Option<String>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            chat_template: Some(DEFAULT_TEMPLATE.to_string()),
            clean_up_tokenization_spaces: false,
            eos_token: "<default_eos_token>".to_string(),
            json_start_token: None,
            bos_token: None,
            unk_token: None,
            sep_token: None,
            pad_token: None,
            cls_token: None,
            mask_token: None,
        }
    }
}

pub fn setup_tokenizer(
    cli_config: &CliConfig,
    config: &LlgTrtConfig,
) -> anyhow::Result<(TokEnv, ChatBuilder)> {
    let tokenizer_folder = cli_config.tokenizer.as_ref().unwrap_or(&cli_config.engine);

    let tokenizer = format!("{}/tokenizer.json", tokenizer_folder);
    log::info!("Loading tokenizer from {:?}", tokenizer);
    let tok_env = toktrie_hf_tokenizers::ByteTokenizerEnv::from_name(&tokenizer, None)?;
    let tok_env: TokEnv = Arc::new(tok_env);
    let trie = tok_env.tok_trie();
    let mut info = trie.info().clone();

    let tok_cfg = &config.tokenizer;
    let toks = tok_env.tokenize_special(&tok_cfg.eos_token);
    ensure!(
        toks.len() == 1,
        "tokenizer_config.json -> eos_token ({:?}) must tokenize to a single token",
        tok_cfg.eos_token
    );
    info.tok_eos = toks[0];

    info.tok_bos = None;
    if let Some(s) = &tok_cfg.bos_token {
        let toks = tok_env.tokenize_special(s);
        ensure!(
            toks.len() == 1,
            "tokenizer_config.json -> bos_token ({:?}) must tokenize to a single token",
            s
        );
        info.tok_bos = Some(toks[0]);
    }

    let tok_env = Arc::new(TokEnvWithTrie::new(tok_env.clone(), trie.with_info(info)));
    let chat_builder = ChatBuilder::new(&tok_cfg)?;

    Ok((tok_env, chat_builder))
}
