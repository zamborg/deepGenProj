# deepGenProj

## TODO (Apr 15):
**Goal is exp v1 done by EOD Wednesday**
- [ ] Take `civil_comments` dataset and replace "text" field with "embedding" field from `gemma2b`
    - get embeddings from [here](https://huggingface.co/docs/transformers/en/main_classes/output)
- [ ] Train DNN using gemma2b as embeddings for [hf-civil-comments](https://huggingface.co/datasets/google/civil_comments)
- [ ] Do the same but for (hf-real-toxicity-prompts)[https://huggingface.co/datasets/allenai/real-toxicity-prompts]
- [ ] Evaluate models on [toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat) human annotated dataset.
  - [ ] For model_input
  - [ ] For model_output
  - [ ] For baseline
     
Per-Person TASKS (TANGIBLE RESULT by April 16 EOD):
- Zubin
  - Train a DNN on RTP / CC data (get this dataloading + training script running)
    - use fake randomiized embeddings FIRST to be ubnlocked
- Justin
  - Get embeddings from gemma2b + (other hf models -- bert tiny)
    - CREATE our embeddding : score dataset (for RTP / CC / toxic-chat)
- Satvik
  - Get evaluation script working for toxic-chat
    - for T5-large model baseline?
    - For some other baseline small toxicity detector model


## RANDOM NOTES: 
STANDARD:

Prompt -> TOX DETECTOR (LLM) -> if tox/nontox:
  Prompt (safety) -> generator (LLM) -> Generation 

INPUT:
  RTP.prompt -> LM -> DNN -> [SCORES] (flirtation... identity_attack_... )
  TARGET: RTP.continuation.{flirtation, identity_ttack, threat, insult, ....}

Prompt -> TokenMatrix -> DecoderLayers (xN) -> 2048 embedding -> Dotproduct w/ Embedding Matrix (100kx2048) -> 100k token_probs

torch.multinomial(preds)
