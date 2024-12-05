# **LLM Response Analyzer** 🚀  
*A tool to unlock the secrets of Large Language Models (LLMs)*  

---

## **Project Overview** 🌟  
The **LLM Response Analyzer** is a comprehensive tool designed to help users explore, understand, and influence the behavior of Large Language Models (LLMs). By tweaking key parameters, visualizing attention mechanisms, and steering the AI’s personality, this project demystifies how LLMs generate their outputs and opens up new possibilities for explainability and control.

---

## **Core Features** 🔍  

1. **Compare Responses with Parameters**  
   - Adjust temperature, top-K, and max tokens to see how model behavior changes.  
   - Compare responses side-by-side to explore creativity, focus, and verbosity.  

2. **🌐 Attention Heatmap Visualization**  
   - Dive into the AI’s thought process by visualizing how it distributes attention across input tokens.  
   - Gain transparency into what the model "pays attention to" during text generation.  

3. **LLM Steering**  
   - Shape the AI’s personality with activation steering.  
   - Example: Transform “I think dogs are…” into:  
     - **Positive:** “the best creatures ever!”  
     - **Negative:** “annoying and loud.”  
     - **Neutral:** “a common household pet.”  

---

## **Why This Project Matters** 🌍  

Understanding AI isn’t just for researchers anymore. This tool bridges the gap by offering:  
- **Transparency:** Visualize how AI makes decisions.  
- **Control:** Shape outputs for tailored use cases (e.g., customer support, creative writing).  
- **Insights:** Learn how parameters like temperature and top-K impact AI creativity and focus.  

---

## **Technology Stack** 🛠️  

| Tool/Technology          | Purpose                           |  
|---------------------------|-----------------------------------|  
| **Python 3.9+**           | Core programming language         |  
| **Streamlit**             | Interactive web interface         |  
| **Hugging Face Transformers** | Model integration and tokenization |  
| **BertViz**               | Attention heatmap visualization   |  
| **Torch**                 | Backend computations and steering |  

---

## **Getting Started** 🚦  

### **Installation**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/rishabhshah13/Adversarial_Patch_XAI590.git
   cd Final Project
   ```  
2. Create a virtual environment and activate it:  
   ```bash
   python3 -m venv env  
   source env/bin/activate  # For Linux/MacOS  
   env\Scripts\activate  # For Windows  
   ```  
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt  
   ```  

### **Run the Application**  
1. Start the backend server:  
   ```bash
   uvicorn backend.api.app:app --reload  
   ```  
2. Run the frontend Streamlit app:  
   ```bash
   streamlit run frontend/app.py  
   ```  

3. Open your browser and visit: [http://localhost:8501](http://localhost:8501)  

---

## **Key Components** 🧩  

### **1. Parameter Tuning**  
Explore how parameters affect responses:  
- **Temperature**: Adjust randomness—low values for safe, predictable text; high values for creative, risky output.  
- **Top-K**: Limit token choices—fewer options lead to stricter focus.  
- **Max Tokens**: Control response length.  

---

### **2. 🌐 Attention Heatmap Visualization**  
- Peek into the model’s brain: See which words the model focuses on while generating a response.  
- Understand why the model generates certain outputs based on its attention patterns.  

---

### **3. LLM Steering**  
- Manipulate internal activations to change the model’s tone or sentiment.  
- Example:  
  - Input: “I think dogs are ”  
  - Steering:  
    - **Love-Hate:** Positive: “incredible and heartwarming.” Negative: “a total waste of time.”  

---

## **Use Cases** 💡  
1. **Creative Writing:** Explore and fine-tune model creativity for story generation.  
2. **Customer Support:** Ensure consistent and empathetic tone.  
3. **Explainability in AI:** Understand why the model produces certain responses.  
4. **AI Debugging:** Identify issues in attention focus or parameter tuning.  

---

## **Credits and Inspiration** ✨  
This project draws inspiration from:  
- [BertViz](https://github.com/jessevig/bertviz): Attention visualization.  
- [TransformerLens](https://github.com/AlignmentResearch/TransformerLens): Internal activation steering.  
- ChatGPT
- [Implementing activation steering](https://www.lesswrong.com/posts/ndyngghzFY388Dnew/implementing-activation-steering#fn0j9pmu8ahchs)
---

## **Future Enhancements** 🚀  
1. **Integration with More Models:** Expand to GPT-4, T5, and others.  
2. **Live Cloud Deployment:** Accessible for non-technical users.  
3. **More Steering Options:** Add personality traits like humor, politeness, etc.  
4. **Enhanced Visualizations:** Animated attention tracking for better insights.  


