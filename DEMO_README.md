# 🔍 Elmstash Demo UI

An interactive Streamlit dashboard showcasing the **Clean Separation Architecture** that separates observation (what happened) from evaluation (how good was it).

## 🚀 Quick Start

### Option 1: Use the Launcher (Recommended)
```bash
python run_demo.py
```

### Option 2: Direct Streamlit
```bash
streamlit run demo_ui.py
```

The demo will open at `http://localhost:8501`

## 📋 Prerequisites

Install demo dependencies:
```bash
pip install -r requirements_demo.txt
```

Or install individually:
```bash
pip install streamlit plotly pandas
```

## 🎯 Demo Features

### 🏗️ Architecture Demo
- **Visual explanation** of the three-layer architecture
- **Code examples** showing how each component works
- **Benefits overview** with clear explanations

### 🔍 Single Analysis Mode
- **Live analysis** of individual input/output pairs
- **Real-time metrics** calculation and display
- **Three-layer breakdown**:
  - 👁️ **Observer**: Objective measurements (entropy, length, patterns)
  - ⚖️ **Evaluator**: Subjective assessments (quality, safety, task completion)
  - 💡 **Insights**: Comprehensive Reports

### 🔄 Batch Comparison Mode
- **Side-by-side model comparison** using the same prompt
- **Radar charts** showing capability differences
- **Detailed metrics tables** for objective comparison

### 📊 Sample Dataset Analysis
- **Batch processing** of multiple samples
- **Distribution analysis** with histograms and scatter plots
- **Correlation analysis** between metrics
- **Quality assessment** across the dataset

## 🎨 UI Components

### Interactive Elements
- **Predefined examples** for quick testing
- **Custom input areas** for your own prompts
- **Model selection** for comparisons
- **Real-time metric updates**

### Visualizations
- **Character frequency distributions**
- **Entropy comparison charts**
- **Model capability radar charts**
- **Quality assessment pie charts**
- **Correlation heatmaps**

### Three-Layer Display
Each analysis shows results from all three architectural layers:

1. **👁️ Observer Tab**: Pure objective measurements
   - Response entropy, token counts, length statistics
   - Pattern detection results
   - No quality judgments

2. **⚖️ Evaluator Tab**: Subjective quality assessments
   - Task completion scores
   - Instruction following ratings
   - Safety assessments
   - Overall quality ratings

3. **💡 Integration Tab**: Combined insights
   - Comprehensive analysis results
   - Actionable recommendations
   - Integrated reporting

## 💡 Demo Script Ideas

### For Technical Audiences
1. **Architecture Overview**: Start with "Architecture Demo" to explain the separation
2. **Single Analysis**: Show how one prompt flows through all three layers
3. **Batch Comparison**: Demonstrate how different models perform on the same task
4. **Flexibility**: Show how the same observed data can be used with different evaluators

### For Business Audiences
1. **Value Proposition**: Start with "Single Analysis" showing immediate insights
2. **Model Comparison**: Show how to objectively compare different AI models
3. **Quality Assessment**: Demonstrate automated quality scoring
4. **Scalability**: Show "Sample Dataset" analysis for batch processing

### Key Demo Points
- **Immediate Value**: Metrics update in real-time as you type
- **Objective vs Subjective**: Clear separation between measurements and judgments
- **Reusability**: Same data works with different evaluation strategies
- **Extensibility**: Easy to add new metrics or evaluation criteria

## 🔧 Customization

### Adding New Examples
Edit the `example_prompts` dictionary in `demo_ui.py`:
```python
example_prompts = {
    "Your Domain": {
        "input": "Your example prompt",
        "response": "Expected model response"
    }
}
```

### Adding New Metrics
1. Add metric calculation in the Observer component
2. Update the UI to display the new metric
3. Add visualization if needed

### Adding New Evaluators
1. Create new evaluator class following the pattern
2. Add to the comparison dropdown
3. Update visualization to show new assessment criteria

## 🐛 Troubleshooting

### Common Issues

**Demo won't start:**
- Check that Streamlit is installed: `pip install streamlit`
- Verify you're in the project root directory
- Check that all dependencies are installed

**Import errors:**
- Make sure the `src` directory is in your Python path
- Verify all required modules are present
- Check that the database can be created (write permissions)

**Slow performance:**
- Reduce dataset size in "Sample Dataset" mode
- Clear browser cache if visualizations aren't updating
- Restart the demo if memory usage gets high

### Getting Help
- Check the main project README for architecture details
- Review the `CLEAN_SEPARATION_ARCHITECTURE.md` for implementation details
- Look at the test files for usage examples

## 🎯 Next Steps After Demo

### For Development
1. **Connect to real APIs**: Replace mock responses with actual model calls
2. **Add more metrics**: Implement semantic similarity, coherence measures
3. **Real-time monitoring**: Add live model performance tracking
4. **Custom evaluators**: Create domain-specific evaluation strategies

### For Production
1. **Scale to React/FastAPI**: When you need more control and polish
2. **Database integration**: Connect to production data sources
3. **User management**: Add authentication and user-specific sessions
4. **API endpoints**: Expose functionality via REST API

The demo provides a solid foundation for understanding the architecture and can be extended based on your specific needs and use cases.