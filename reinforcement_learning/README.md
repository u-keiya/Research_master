# settings
## Phi-3-mini-instruct
```
ppo_config = PPOConfig(
    model_name="/path/to/model",
    learning_rate=5e-6,
    ppo_epochs=4,
    batch_size=8,
    mini_batch_size=2,
    gradient_accumulation_steps=4,
    adap_kl_ctrl=False,
    init_kl_coef=0.01,
    remove_unused_columns=False,
    kl_penalty="full"
    )
```

## Zamba2-2.7B-instruct
```
ppo_config = PPOConfig(
    model_name="/path/to/Zamba2-2.7B-instruct",
    learning_rate=2e-6,
    ppo_epochs=4,
    batch_size=8,
    mini_batch_size=2,
    gradient_accumulation_steps=4,
    adap_kl_ctrl=False,
    init_kl_coef=0.005,
    remove_unused_columns=False,
    kl_penalty="full",
    score_clip=2
    )
```