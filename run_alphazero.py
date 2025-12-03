from learning.training_loop import AlphaZeroTrainer

if __name__ == "__main__":
    print("Initializing AlphaZero Training Pipeline...")
    
    # This will load config.json, setup the environment, and run the self-play loop
    trainer = AlphaZeroTrainer(config_path="config.json")
    
    # Start Training
    trainer.run_training_loop()