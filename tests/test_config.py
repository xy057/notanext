from notagen.config import PRESETS, ModelConfig

RELEASED = {
    "weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_12_c_layers_3_h_size_768_lr_0.0002_batch_8.pth": "small",
    "weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_16_c_layers_3_h_size_1024_lr_0.0001_batch_4.pth": "medium",
    "weights_notagen_pretrain_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_0.0001_batch_4.pth": "large",
    "weights_notagen_pretrain-finetune_p_size_16_p_length_1024_p_layers_c_layers_6_20_h_size_1280_lr_1e-05_batch_1.pth": "large",
    "weights_notagen_pretrain-finetune-RL3_beta_0.1_lambda_10_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-06_batch_1.pth": "large",
    "weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth": "x",
}


def test_released_names_parse():
    for name, preset in RELEASED.items():
        assert ModelConfig.from_legacy_name(name) == PRESETS[preset], name
