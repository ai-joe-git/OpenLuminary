import os
from typing import Dict, List, Optional, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Qwen3Interface:
    """Interface to Qwen3 model for financial analysis."""
    
    def __init__(
        self, 
        model_id: str = "Qwen/Qwen3-235B-A22B", 
        use_thinking_mode: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.use_thinking_mode = use_thinking_mode
        
        print(f"Loading Qwen3 model from {model_id}...")
        # In production, you would use the full model, but for testing:
        # Consider using a smaller model or quantized version initially
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("Model loaded successfully!")
        
    def analyze_financial_data(
        self, 
        data: Dict[str, Any], 
        analysis_type: str,
        prompt_template: Optional[str] = None
    ) -> str:
        """
        Analyze financial data using Qwen3.
        
        Args:
            data: Financial data dictionary
            analysis_type: Type of analysis to perform (risk, optimization, prediction)
            prompt_template: Custom prompt template to use
            
        Returns:
            Analysis results as a string
        """
        if prompt_template is None:
            prompt_template = self._get_default_prompt_template(analysis_type)
            
        # Format the financial data as a structured input
        formatted_data = self._format_data_for_prompt(data)
        
        # Create the full prompt
        prompt = prompt_template.format(data=formatted_data)
        
        # Add thinking mode toggle if needed
        if self.use_thinking_mode:
            prompt = "/think\n" + prompt
        
        # Generate response from Qwen3
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.2,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split(prompt)[-1].strip()
    
    def _get_default_prompt_template(self, analysis_type: str) -> str:
        """Get default prompt template based on analysis type."""
        templates = {
            "risk": (
                "You are a financial risk analyst. Analyze the following financial data "
                "and provide a comprehensive risk assessment, including VaR (Value at Risk), "
                "potential drawdowns, and risk factors.\n\n{data}"
            ),
            "optimization": (
                "You are a portfolio optimization specialist. Analyze the following financial data "
                "and provide optimal portfolio allocations based on modern portfolio theory. "
                "Include expected returns, volatility, and Sharpe ratio.\n\n{data}"
            ),
            "prediction": (
                "You are a market analyst. Based on the following financial data, "
                "provide market predictions and trend analysis. Identify key indicators "
                "and potential market movements.\n\n{data}"
            )
        }
        return templates.get(analysis_type, templates["risk"])
    
    def _format_data_for_prompt(self, data: Dict[str, Any]) -> str:
        """Format financial data as a structured string for the prompt."""
        result = []
        for key, value in data.items():
            if isinstance(value, dict):
                formatted_value = "\n  " + "\n  ".join(f"{k}: {v}" for k, v in value.items())
                result.append(f"{key}:{formatted_value}")
            elif isinstance(value, list):
                formatted_value = "\n  - " + "\n  - ".join(str(item) for item in value)
                result.append(f"{key}:{formatted_value}")
            else:
                result.append(f"{key}: {value}")
        
        return "\n".join(result)
