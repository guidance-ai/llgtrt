use crate::{chat::ChatBuilder, constraint_mgr::ConstraintMgr};
use toktrie::{TokEnv, TokenId};

// there's generally an Arc() around this
pub struct AppState {
    pub tok_env: TokEnv,
    pub tok_bos: Option<TokenId>,
    pub tok_eos_chat: Option<TokenId>,
    pub tok_eos_completions: Option<TokenId>,
    pub next_client_req_id: std::sync::atomic::AtomicUsize,
    pub chat_builder: ChatBuilder,
    pub constraint_mgr: ConstraintMgr,
}

impl AppState {
    pub fn tokenize_with_bos(&self, s: &str) -> Vec<TokenId> {
        let mut tokens = self.tok_env.tokenize(s);
        let trie = self.tok_env.tok_trie();
        if let Some(bos) = trie.info().tok_bos {
            if tokens.len() == 0 || tokens[0] != bos {
                tokens.insert(0, bos);
            }
        }
        tokens
    }
}
