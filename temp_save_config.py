from probablyprofit.config import get_config, save_config

config = get_config()
save_config(config)
print("Configuration saved to ~/.probablyprofit/config.yaml")
