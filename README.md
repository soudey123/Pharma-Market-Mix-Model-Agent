# Pharma Market Mix Model Agent

An intelligent agent for pharmaceutical marketing teams to analyze and optimize marketing spend across channels using Bayesian Market Mix Modeling (MMM).

## Features

### Marketing Analytics
- **6 Pharma-Specific Channels**: DTC TV, DTC Digital, HCP Detailing, Conferences, Samples, Journal Ads
- **Adstock Transformations**: Geometric and Weibull decay with pharma-appropriate 4-12 week carryover
- **Saturation Curves**: Hill function modeling for diminishing returns
- **Channel Contribution Decomposition**: Waterfall charts and attribution analysis

### AI-Powered Insights
- **Natural Language Q&A**: Ask questions about your results using Claude API
- **Automated Summaries**: Executive summaries and recommendations
- **Report Generation**: Comprehensive analysis reports

### Budget Optimization
- **Constrained Optimization**: Respect channel-specific budget limits
- **Scenario Analysis**: Compare different budget allocation strategies
- **Marginal ROI Curves**: Visualize diminishing returns by channel

### Integration Ready
- **Streamlit UI**: Interactive web application
- **FastAPI Endpoints**: REST API for automation
- **n8n Workflow Templates**: Ready-to-use workflow automation

## Quick Start

### Prerequisites

- Python 3.10+
- Anthropic API key (for AI insights)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/pharma-mmm-agent.git
cd pharma-mmm-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Running the Application

#### Streamlit UI
```bash
streamlit run app.py
```
Access at http://localhost:8501

#### FastAPI Server
```bash
python api.py
# or
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```
- API docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Generate Sample Data

```python
from src.utils.data_generator import generate_sample_data

data, true_effects = generate_sample_data("data/sample_pharma_data.csv", n_weeks=156)
print(f"Generated {len(data)} weeks of data")
```

## Project Structure

```
pharma-mmm-agent/
├── app.py                  # Streamlit application
├── api.py                  # FastAPI application
├── requirements.txt        # Python dependencies
├── src/
│   ├── config.py          # Configuration settings
│   ├── services/          # Stateless service layer
│   │   ├── data_service.py        # Data ingestion & validation
│   │   ├── modeling_service.py    # Bayesian MMM with PyMC
│   │   ├── optimization_service.py # Budget optimization
│   │   └── insight_service.py     # Claude-powered insights
│   ├── ui/
│   │   └── pages/         # Streamlit page modules
│   └── utils/
│       └── data_generator.py      # Sample data generation
├── data/                   # Data directory
├── n8n_workflows/          # n8n workflow templates
│   └── pharma_mmm_workflow.json
└── tests/                  # Test suite
```

## API Reference

### Data Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/data/validate` | POST | Validate data for MMM requirements |
| `/api/v1/data/summary` | POST | Get data summary statistics |
| `/api/v1/data/preprocess` | POST | Preprocess data for modeling |
| `/api/v1/data/adstock` | POST | Apply adstock transformation |

### Model Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/model/train` | POST | Train Bayesian MMM (sync) |
| `/api/v1/model/train/async` | POST | Train model asynchronously |
| `/api/v1/model/status/{job_id}` | GET | Check training status |
| `/api/v1/model/predict` | POST | Make predictions |
| `/api/v1/model/{model_id}/decomposition` | GET | Get sales decomposition |

### Optimization Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/optimize/budget` | POST | Optimize budget allocation |
| `/api/v1/optimize/scenarios` | POST | Analyze multiple scenarios |
| `/api/v1/optimize/marginal-roi` | POST | Get marginal ROI curves |

### Insight Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/insights/generate` | POST | Generate AI insights |
| `/api/v1/insights/qa` | POST | Answer questions |
| `/api/v1/insights/report` | POST | Generate full report |

## n8n Integration Guide

### Overview

This application is designed for seamless integration with n8n workflows. Each service endpoint follows REST conventions and returns JSON responses compatible with n8n's HTTP Request node.

### Importing the Workflow

1. Open your n8n instance
2. Go to Workflows → Import from File
3. Select `n8n_workflows/pharma_mmm_workflow.json`
4. Update the API URL if not running on localhost:8000

### Workflow Structure

The included workflow demonstrates a complete MMM analysis pipeline:

```
[Manual Trigger]
      ↓
[Generate Sample Data]  → Creates test data
      ↓
[Validate Data]         → Checks data quality
      ↓
[Is Valid?]            → Conditional routing
      ↓
[Train MMM Model]      → Bayesian modeling (5-10 min)
      ↓
[Optimize Budget]      → Budget allocation
      ↓
[Generate AI Insights] → Claude-powered analysis
      ↓
[Format Report]        → Final output
```

### Building Custom Workflows

#### Example: Scheduled Weekly Analysis

```json
{
  "nodes": [
    {
      "type": "n8n-nodes-base.scheduleTrigger",
      "parameters": {
        "rule": {
          "interval": [{"field": "weeks", "weeksInterval": 1}]
        }
      }
    },
    {
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "http://your-server:8000/api/v1/model/train",
        "body": {
          "data_json": "{{ $json.data }}",
          "spend_columns": ["dtc_tv_spend", "hcp_detailing_spend"]
        }
      }
    }
  ]
}
```

#### Example: Slack Integration

Add a Slack node after insights generation:

```json
{
  "type": "n8n-nodes-base.slack",
  "parameters": {
    "channel": "#marketing-analytics",
    "text": "Weekly MMM Analysis Complete\n\nR²: {{ $json.r_squared }}\nTop Channel: {{ $json.top_channel }}"
  }
}
```

### Converting Services to n8n Nodes

Each service in `src/services/` is designed as a stateless function with Pydantic input/output contracts. To convert to a custom n8n node:

1. **Create Node Definition**
```typescript
// nodes/PharmaMmm/PharmaMmm.node.ts
export class PharmaMmm implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'Pharma MMM',
    name: 'pharmaMmm',
    group: ['transform'],
    version: 1,
    description: 'Market Mix Modeling for Pharma',
    inputs: ['main'],
    outputs: ['main'],
    properties: [
      {
        displayName: 'Operation',
        name: 'operation',
        type: 'options',
        options: [
          { name: 'Train Model', value: 'train' },
          { name: 'Optimize Budget', value: 'optimize' },
          { name: 'Generate Insights', value: 'insights' }
        ]
      }
    ]
  };
}
```

2. **Map Service Methods**
```typescript
async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
  const operation = this.getNodeParameter('operation', 0) as string;

  // Call your FastAPI endpoint
  const response = await this.helpers.httpRequest({
    method: 'POST',
    url: `${baseUrl}/api/v1/${operation}`,
    body: inputData
  });

  return [this.helpers.returnJsonArray(response)];
}
```

## Pharma-Specific Considerations

### Channel Characteristics

| Channel | Typical Decay | Carryover | Notes |
|---------|---------------|-----------|-------|
| DTC TV | 0.5-0.85 | 10 weeks | High reach, brand building |
| DTC Digital | 0.3-0.7 | 6 weeks | Targetable, measurable |
| HCP Detailing | 0.6-0.9 | 12 weeks | High influence, relationship-based |
| Conferences | 0.4-0.8 | 8 weeks | Seasonal, KOL engagement |
| Samples | 0.5-0.85 | 10 weeks | Direct trial, prescription starter |
| Journal Ads | 0.4-0.75 | 8 weeks | Credibility, professional audience |

### Model Assumptions

1. **Longer Carryover**: Pharma marketing effects persist longer than CPG (4-12 weeks vs 1-4 weeks)
2. **Regulatory Constraints**: DTC advertising has specific requirements affecting creative rotation
3. **HCP Influence**: Physician recommendations drive significant prescription volume
4. **Seasonality**: Q1 typically lower (insurance resets), Q4 higher (end-of-year push)

## Configuration

### Environment Variables

```bash
# Required for AI insights
ANTHROPIC_API_KEY=your-api-key

# Optional
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_PORT=8501
```

### Model Configuration

Edit `src/config.py` to adjust:

```python
class ModelConfig:
    adstock_max_lag: int = 12  # Maximum carryover weeks
    n_samples: int = 2000      # MCMC samples
    n_tune: int = 1000         # Tuning iterations
    n_chains: int = 2          # Parallel chains
    target_accept: float = 0.9 # NUTS acceptance rate
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

```bash
# Format
black src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/
```

## Troubleshooting

### Common Issues

**Model training is slow**
- Reduce `n_samples` and `n_tune` for faster iteration
- Use `n_chains=1` during development
- Consider using GPU acceleration with JAX backend

**Memory errors**
- Reduce dataset size for initial testing
- Use async training endpoint for large models

**n8n connection refused**
- Ensure API server is running
- Check firewall/network settings
- Verify URL in n8n HTTP Request node

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/your-org/pharma-mmm-agent/issues)
- Documentation: See `/docs` endpoint when API is running
