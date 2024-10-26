use crate::{chat::ChatBuilder, config::Config};
use anyhow::{anyhow, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, sync::Arc};
use toktrie::{TokEnv, TokEnvWithTrie};

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenizerConfig {
    #[serde(default)]
    pub added_tokens_decoder: HashMap<String, TokenProperties>,

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
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenProperties {
    pub content: String,
    #[serde(default)]
    pub lstrip: bool,
    #[serde(default)]
    pub normalized: bool,
    #[serde(default)]
    pub rstrip: bool,
    #[serde(default)]
    pub single_word: bool,
    #[serde(default)]
    pub special: bool,
}

pub fn setup_tokenizer(cli_config: &Config) -> anyhow::Result<(TokEnv, ChatBuilder)> {
    let tokenizer_folder = cli_config.tokenizer.as_ref().unwrap_or(&cli_config.engine);
    let tokenizer_config = format!("{}/tokenizer_config.json", tokenizer_folder);
    log::info!("Loading tokenizer config from {:?}", tokenizer_config);
    let mut tok_cfg: TokenizerConfig =
        serde_json::from_reader(std::fs::File::open(tokenizer_config)?)
            .map_err(|e| anyhow!("error loading tokenizer_config.json: {}", e))?;

    let tokenizer_config_llg = format!("{}/tokenizer_config_llgtrt.json", tokenizer_folder);
    log::info!("Checking for overrides in {:?}", tokenizer_config_llg);
    if std::fs::exists(&tokenizer_config_llg)? {
        let mut json = serde_json::to_value(&tok_cfg)?;
        let overrides: Value = serde_json::from_reader(std::fs::File::open(tokenizer_config_llg)?)
            .map_err(|e| anyhow!("JSON error in tokenizer_config_llgtrt.json: {}", e))?;
        for (k, v) in overrides.as_object().expect("overrides must be an object") {
            if v.is_null() {
                continue;
            }
            json.as_object_mut().unwrap().insert(k.clone(), v.clone());
        }
        tok_cfg = serde_json::from_value(json)
            .map_err(|e| anyhow!("error applying tokenizer_config_llgtrt.json: {}", e))?;
    }

    let chat_template = format!("{}/chat_template.j2", tokenizer_folder);
    log::info!("Checking for separate chat template in {:?}", chat_template);
    if std::fs::exists(&chat_template)? {
        tok_cfg.chat_template = Some(std::fs::read_to_string(chat_template)?);
    }

    let tokenizer = format!("{}/tokenizer.json", tokenizer_folder);
    log::info!("Loading tokenizer from {:?}", tokenizer);
    let tok_env = toktrie_hf_tokenizers::ByteTokenizerEnv::from_name(&tokenizer, None)?;
    let tok_env: TokEnv = Arc::new(tok_env);
    let trie = tok_env.tok_trie();
    let mut info = trie.info().clone();

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
