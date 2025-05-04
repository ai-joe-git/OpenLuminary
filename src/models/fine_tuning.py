import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Qwen3FineTuner:
    """Fine-tune Qwen3 model on financial data."""
    
    def __init__(
        self, 
        model_id: str = "Qwen/Qwen3-7B-A1B",  # Start with smaller model for fine-tuning
        output_dir: str = "./fine_tuned_model",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_id = model_id
        self.output_dir = output_dir
        self.device = device
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load the base model for fine-tuning."""
        logger.info(f"Loading base model {self.model_id}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        logger.info("Base model loaded successfully")
        
    def prepare_financial_dataset(self, data_path: str) -> Dataset:
        """
        Prepare financial dataset for fine-tuning.
        
        Args:
            data_path: Path to CSV file containing financial data and prompts
            
        Returns:
            HuggingFace Dataset ready for fine-tuning
        """
        logger.info(f"Preparing dataset from {data_path}")
        
        # Load data from CSV
        df = pd.read_csv(data_path)
        
        # Ensure required columns exist
        required_cols = ["instruction", "response"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Dataset must contain columns: {required_cols}")
        
        # Format data for instruction fine-tuning
        def format_instruction(row):
            return f"<|im_start|>user\n{row['instruction']}<|im_end|>\n<|im_start|>assistant\n{row['response']}<|im_end|>"
        
        df["text"] = df.apply(format_instruction, axis=1)
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_pandas(df[["text"]])
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=2048)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        logger.info(f"Dataset prepared with {len(tokenized_dataset)} examples")
        
        return tokenized_dataset
    
    def fine_tune(
        self, 
        dataset: Dataset, 
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        save_steps: int = 500
    ):
        """
        Fine-tune the model on financial data.
        
        Args:
            dataset: Prepared dataset for fine-tuning
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            save_steps: Save checkpoint every N steps
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        logger.info("Starting fine-tuning process")
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            save_steps=save_steps,
            logging_steps=100,
            learning_rate=learning_rate,
            weight_decay=0.01,
            fp16=True,
            report_to="tensorboard",
            save_total_limit=2,
        )
        
        # Create Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        logger.info("Training started")
        trainer.train()
        
        # Save the fine-tuned model
        logger.info(f"Saving fine-tuned model to {self.output_dir}")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info("Fine-tuning completed successfully")

    def create_synthetic_financial_data(
        self, 
        output_path: str = "data/synthetic_financial_data.csv",
        num_examples: int = 100
    ):
        """
        Create synthetic financial data for fine-tuning when real data is not available.
        
        Args:
            output_path: Path to save the synthetic data
            num_examples: Number of examples to generate
        """
        logger.info(f"Generating {num_examples} synthetic financial examples")
        
        # Define templates for financial analysis tasks
        templates = [
            {
                "task": "portfolio_optimization",
                "instruction_template": "Optimize a portfolio with the following assets and constraints:\n{assets}\n{constraints}\nMaximize the Sharpe ratio while maintaining a risk level below {risk_level}.",
                "response_template": "Based on the provided assets and constraints, I recommend the following portfolio allocation:\n{allocation}\n\nThis allocation yields:\n- Expected annual return: {return_rate}\n- Annual volatility: {volatility}\n- Sharpe ratio: {sharpe}\n\nRationale: {rationale}"
            },
            {
                "task": "risk_assessment",
                "instruction_template": "Assess the risk of the following portfolio:\n{portfolio}\n\nProvide Value at Risk (VaR), Conditional VaR, and identify key risk factors.",
                "response_template": "Risk Assessment for the portfolio:\n\n1. Value at Risk (95% confidence):\n   - 1-day VaR: {var_1d}\n   - 10-day VaR: {var_10d}\n\n2. Conditional VaR (Expected Shortfall):\n   - 1-day CVaR: {cvar_1d}\n   - 10-day CVaR: {cvar_10d}\n\n3. Key Risk Factors:\n{risk_factors}\n\n4. Stress Test Results:\n{stress_test}\n\nRecommendations: {recommendations}"
            },
            {
                "task": "market_prediction",
                "instruction_template": "Based on the following market data and economic indicators, predict the likely movement of {asset} over the next {timeframe}:\n{market_data}\n{economic_indicators}",
                "response_template": "Market Prediction for {asset} over the next {timeframe}:\n\nPrediction: {prediction}\n\nKey factors supporting this prediction:\n{factors}\n\nPotential catalysts to monitor:\n{catalysts}\n\nRisk factors that could alter this outlook:\n{risks}\n\nConfidence level: {confidence}"
            }
        ]
        
        # Generate synthetic data
        data = []
        for i in range(num_examples):
            template = templates[i % len(templates)]
            
            if template["task"] == "portfolio_optimization":
                assets = "\n".join([
                    f"- Asset {j+1}: Expected return {round(0.05 + 0.1 * torch.rand(1).item(), 4)}, volatility {round(0.1 + 0.2 * torch.rand(1).item(), 4)}"
                    for j in range(5)
                ])
                constraints = "\n".join([
                    "- Maximum allocation per asset: 40%",
                    "- Minimum allocation per asset: 5%",
                    f"- Sector constraints: Technology max {30 + int(20 * torch.rand(1).item())}%, Finance max {30 + int(20 * torch.rand(1).item())}%"
                ])
                risk_level = f"{round(0.15 + 0.1 * torch.rand(1).item(), 4)}"
                
                instruction = template["instruction_template"].format(
                    assets=assets,
                    constraints=constraints,
                    risk_level=risk_level
                )
                
                allocation = "\n".join([
                    f"- Asset {j+1}: {round(0.1 + 0.3 * torch.rand(1).item(), 4) * 100}%"
                    for j in range(5)
                ])
                return_rate = f"{round(0.08 + 0.06 * torch.rand(1).item(), 4) * 100}%"
                volatility = f"{round(0.12 + 0.08 * torch.rand(1).item(), 4) * 100}%"
                sharpe = f"{round(0.8 + 1.2 * torch.rand(1).item(), 2)}"
                rationale = "This allocation balances risk and return by diversifying across assets while maintaining exposure to growth opportunities. The overweight in Asset 3 is due to its favorable risk-return profile and lower correlation with other assets in the portfolio."
                
                response = template["response_template"].format(
                    allocation=allocation,
                    return_rate=return_rate,
                    volatility=volatility,
                    sharpe=sharpe,
                    rationale=rationale
                )
            
            elif template["task"] == "risk_assessment":
                portfolio = "\n".join([
                    f"- {['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'BAC', 'WMT', 'PG'][j % 10]}: {round(0.05 + 0.15 * torch.rand(1).item(), 4) * 100}%"
                    for j in range(8)
                ])
                
                instruction = template["instruction_template"].format(
                    portfolio=portfolio
                )
                
                var_1d = f"{round(0.015 + 0.01 * torch.rand(1).item(), 4) * 100}%"
                var_10d = f"{round(0.04 + 0.03 * torch.rand(1).item(), 4) * 100}%"
                cvar_1d = f"{round(0.025 + 0.015 * torch.rand(1).item(), 4) * 100}%"
                cvar_10d = f"{round(0.06 + 0.04 * torch.rand(1).item(), 4) * 100}%"
                
                risk_factors = "\n".join([
                    "- High technology sector concentration (62% of portfolio)",
                    "- Interest rate sensitivity due to growth stock focus",
                    "- Limited international diversification (87% US exposure)",
                    "- Momentum factor exposure creating potential volatility"
                ])
                
                stress_test = "\n".join([
                    "- Market crash scenario (-20% broad market): Portfolio impact -24.3%",
                    "- Tech sector correction (-15%): Portfolio impact -18.7%",
                    "- Interest rate hike (100bps): Portfolio impact -7.2%",
                    "- Inflation surge (2% above expectations): Portfolio impact -5.8%"
                ])
                
                recommendations = "Consider increasing diversification by adding exposure to value stocks, international markets, and alternative assets. Implementing a 10% allocation to defensive sectors could reduce portfolio VaR by approximately 15%."
                
                response = template["response_template"].format(
                    var_1d=var_1d,
                    var_10d=var_10d,
                    cvar_1d=cvar_1d,
                    cvar_10d=cvar_10d,
                    risk_factors=risk_factors,
                    stress_test=stress_test,
                    recommendations=recommendations
                )
            
            else:  # market_prediction
                asset = ["S&P 500", "NASDAQ", "Bitcoin", "EUR/USD", "Gold", "Crude Oil", "10-Year Treasury", "Tesla Stock"][i % 8]
                timeframe = ["month", "quarter", "6 months", "year"][i % 4]
                
                market_data = "\n".join([
                    f"- Current {asset} price: ${round(100 + 900 * torch.rand(1).item(), 2)}",
                    f"- 50-day moving average: ${round(100 + 900 * torch.rand(1).item(), 2)}",
                    f"- 200-day moving average: ${round(100 + 900 * torch.rand(1).item(), 2)}",
                    f"- RSI: {round(30 + 40 * torch.rand(1).item(), 1)}",
                    f"- Trading volume: {round(0.8 + 0.4 * torch.rand(1).item(), 2)}x average"
                ])
                
                economic_indicators = "\n".join([
                    f"- GDP Growth: {round(1.5 + 3 * torch.rand(1).item(), 1)}%",
                    f"- Inflation Rate: {round(2 + 4 * torch.rand(1).item(), 1)}%",
                    f"- Unemployment: {round(3 + 3 * torch.rand(1).item(), 1)}%",
                    f"- Federal Funds Rate: {round(2 + 3 * torch.rand(1).item(), 1)}%",
                    f"- Consumer Sentiment Index: {round(70 + 30 * torch.rand(1).item(), 1)}"
                ])
                
                instruction = template["instruction_template"].format(
                    asset=asset,
                    timeframe=timeframe,
                    market_data=market_data,
                    economic_indicators=economic_indicators
                )
                
                prediction = ["bullish with an expected increase of 8-12%", 
                              "moderately bullish with an expected increase of 4-7%",
                              "neutral with sideways movement expected (-2% to +3%)",
                              "moderately bearish with an expected decrease of 3-6%",
                              "bearish with an expected decrease of 7-12%"][i % 5]
                
                factors = "\n".join([
                    "1. Technical indicators show momentum with price above both 50-day and 200-day moving averages",
                    "2. Economic growth remains resilient despite inflation concerns",
                    "3. Institutional investor positioning shows increasing allocation",
                    "4. Relative valuation metrics are within historical norms"
                ])
                
                catalysts = "\n".join([
                    "1. Upcoming Federal Reserve policy meeting",
                    "2. Q2 earnings reports expected to exceed analyst expectations",
                    "3. Potential fiscal stimulus package announcement",
                    "4. Resolution of ongoing supply chain constraints"
                ])
                
                risks = "\n".join([
                    "1. Inflation persisting above central bank targets",
                    "2. Geopolitical tensions escalating in Eastern Europe or Asia",
                    "3. Unexpected shift in monetary policy stance",
                    "4. Deterioration in consumer sentiment indicators"
                ])
                
                confidence = f"{round(60 + 30 * torch.rand(1).item(), 0)}%"
                
                response = template["response_template"].format(
                    asset=asset,
                    timeframe=timeframe,
                    prediction=prediction,
                    factors=factors,
                    catalysts=catalysts,
                    risks=risks,
                    confidence=confidence
                )
            
            data.append({
                "instruction": instruction,
                "response": response
            })
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(data).to_csv(output_path, index=False)
        logger.info(f"Synthetic data saved to {output_path}")
        
        return output_path
