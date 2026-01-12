# Pharma MMM Agent - Architecture & Workflow Diagrams

## High-Level Architecture

```mermaid
flowchart TB
    subgraph Users["👥 Users"]
        MktTeam["Marketing Team"]
        Analyst["Data Analyst"]
        Automation["n8n Workflows"]
    end

    subgraph UI["🖥️ Streamlit UI (app.py)"]
        Home["🏠 Home"]
        DataPage["📊 Data Overview"]
        ModelPage["🧮 Model Training"]
        ResultsPage["📈 Results Dashboard"]
        OptPage["💰 Budget Optimization"]
        ChatPage["💬 Chat Interface"]
    end

    subgraph API["🔌 FastAPI (api.py)"]
        DataAPI["/api/v1/data/*"]
        ModelAPI["/api/v1/model/*"]
        OptAPI["/api/v1/optimize/*"]
        InsightAPI["/api/v1/insights/*"]
    end

    subgraph Agents["🤖 Agent Layer"]
        DataAgent["Data Agent"]
        ModelAgent["Modeling Agent"]
        OptAgent["Optimization Agent"]
        InsightAgent["Insight Agent"]
    end

    subgraph Services["⚙️ Services Layer"]
        DataSvc["DataService"]
        ModelSvc["ModelingService"]
        OptSvc["OptimizationService"]
        InsightSvc["InsightService"]
    end

    subgraph External["🌐 External"]
        Claude["Claude API"]
        PyMC["PyMC/Bayesian"]
    end

    MktTeam --> UI
    Analyst --> UI
    Automation --> API

    UI --> Agents
    API --> Services

    Agents --> Services

    DataAgent --> DataSvc
    ModelAgent --> ModelSvc
    OptAgent --> OptSvc
    InsightAgent --> InsightSvc

    InsightSvc --> Claude
    ModelSvc --> PyMC
```

## User Journey Flow

```mermaid
flowchart LR
    subgraph Step1["1️⃣ Data Ingestion"]
        Upload["Upload CSV/Excel"]
        Generate["Generate Sample Data"]
        Validate["Validate Data"]
    end

    subgraph Step2["2️⃣ Data Exploration"]
        Summary["View Summary Stats"]
        TimeSeries["Analyze Time Series"]
        Correlations["Check Correlations"]
    end

    subgraph Step3["3️⃣ Model Training"]
        Config["Configure Model"]
        Train["Train Bayesian MMM"]
        Diagnostics["Check Diagnostics"]
    end

    subgraph Step4["4️⃣ Results Analysis"]
        Contributions["Channel Contributions"]
        ROI["ROI Analysis"]
        ResponseCurves["Response Curves"]
    end

    subgraph Step5["5️⃣ Optimization"]
        SetBudget["Set Total Budget"]
        Constraints["Define Constraints"]
        Optimize["Run Optimization"]
    end

    subgraph Step6["6️⃣ Insights"]
        QA["Ask Questions"]
        Summary2["Get Summaries"]
        Report["Generate Reports"]
    end

    Step1 --> Step2 --> Step3 --> Step4 --> Step5 --> Step6
```

## Streamlit UI Pages

```mermaid
flowchart TB
    subgraph App["📱 Streamlit Application"]
        direction TB

        subgraph HomePage["🏠 Home Page"]
            Welcome["Welcome & Overview"]
            QuickStart["Getting Started Guide"]
            Features["Feature Highlights"]
        end

        subgraph DataPage["📊 Data Overview Page"]
            direction LR
            Upload2["File Upload"]
            Validation["Data Validation"]
            Preview["Data Preview"]
            Viz["Visualizations"]
        end

        subgraph ModelPage["🧮 Model Training Page"]
            direction LR
            AdstockConfig["Adstock Settings"]
            SatConfig["Saturation Settings"]
            MCMCConfig["MCMC Settings"]
            TrainBtn["Train Model"]
        end

        subgraph ResultsPage["📈 Results Dashboard"]
            direction LR
            Waterfall["Waterfall Chart"]
            ContribPie["Contribution Pie"]
            ROIBars["ROI by Channel"]
            Coefficients["Coefficients"]
        end

        subgraph OptPage["💰 Budget Optimization"]
            direction LR
            CurrentAlloc["Current Allocation"]
            Constraints2["Set Constraints"]
            OptResults["Optimized Results"]
            Scenarios["Scenario Analysis"]
        end

        subgraph ChatPage["💬 Chat Interface"]
            direction LR
            QuickActions["Quick Actions"]
            ChatHistory["Chat History"]
            QAInput["Question Input"]
        end
    end

    HomePage --> DataPage
    DataPage --> ModelPage
    ModelPage --> ResultsPage
    ResultsPage --> OptPage
    OptPage --> ChatPage
```

## Services Layer Detail

```mermaid
flowchart TB
    subgraph DataService["📁 DataService"]
        Ingest["ingest_data()"]
        Validate2["validate_data()"]
        GetSummary["get_data_summary()"]
        Preprocess["preprocess_data()"]
        Adstock["apply_adstock_transform()"]
        Saturation["apply_saturation()"]
    end

    subgraph ModelingService["🧮 ModelingService"]
        TrainModel["train_model()"]
        GetDecomp["get_decomposition()"]
        Predict["predict()"]
    end

    subgraph OptimizationService["💰 OptimizationService"]
        OptBudget["optimize_budget()"]
        AnalyzeScenarios["analyze_scenarios()"]
        MarginalROI["get_marginal_roi_curve()"]
    end

    subgraph InsightService["💡 InsightService"]
        GenInsight["generate_insight()"]
        AnswerQ["answer_question()"]
        GenReport["generate_report()"]
    end

    Ingest --> Validate2
    Validate2 --> GetSummary
    GetSummary --> Preprocess
    Preprocess --> Adstock
    Adstock --> Saturation

    Saturation --> TrainModel
    TrainModel --> GetDecomp
    TrainModel --> Predict

    GetDecomp --> OptBudget
    OptBudget --> AnalyzeScenarios
    AnalyzeScenarios --> MarginalROI

    GetDecomp --> GenInsight
    OptBudget --> GenInsight
    GenInsight --> AnswerQ
    AnswerQ --> GenReport
```

## Bayesian MMM Model Flow

```mermaid
flowchart TB
    subgraph Input["📥 Input Data"]
        RawSpend["Raw Spend Data"]
        Sales["Sales Data"]
        Controls["Control Variables"]
    end

    subgraph Transform["🔄 Transformations"]
        AdstockT["Adstock Transform"]
        SaturationT["Saturation (Hill)"]
        Normalize["Normalization"]
    end

    subgraph Model["🎯 PyMC Model"]
        Priors["Define Priors"]
        Likelihood["Likelihood Function"]
        MCMC["MCMC Sampling"]
    end

    subgraph Output["📤 Outputs"]
        Posterior["Posterior Distributions"]
        Coefficients2["Channel Coefficients"]
        Decay["Decay Rates"]
        Fitted["Fitted Values"]
    end

    RawSpend --> AdstockT
    AdstockT --> SaturationT
    SaturationT --> Normalize
    Sales --> Normalize
    Controls --> Normalize

    Normalize --> Priors
    Priors --> Likelihood
    Likelihood --> MCMC

    MCMC --> Posterior
    Posterior --> Coefficients2
    Posterior --> Decay
    Posterior --> Fitted
```

## Adstock & Saturation Transformations

```mermaid
flowchart LR
    subgraph Adstock["📉 Adstock (Carryover)"]
        direction TB
        Raw["Raw Spend: $100K"]
        Week1["Week 1: $100K"]
        Week2["Week 2: $70K (decay=0.7)"]
        Week3["Week 3: $49K"]
        Week4["Week 4: $34K"]

        Raw --> Week1 --> Week2 --> Week3 --> Week4
    end

    subgraph Saturation["📈 Saturation (Hill Function)"]
        direction TB
        Low["Low Spend → High Marginal Return"]
        Med["Medium Spend → Moderate Return"]
        High["High Spend → Diminishing Return"]

        Low --> Med --> High
    end

    Adstock --> Saturation
```

## FastAPI Endpoints

```mermaid
flowchart TB
    subgraph Endpoints["🔌 API Endpoints"]
        subgraph Data["/api/v1/data"]
            D1["POST /validate"]
            D2["POST /summary"]
            D3["POST /preprocess"]
            D4["POST /adstock"]
        end

        subgraph Model["/api/v1/model"]
            M1["POST /train"]
            M2["POST /train/async"]
            M3["GET /status/{job_id}"]
            M4["POST /predict"]
            M5["GET /{model_id}/decomposition"]
        end

        subgraph Optimize["/api/v1/optimize"]
            O1["POST /budget"]
            O2["POST /scenarios"]
            O3["POST /marginal-roi"]
        end

        subgraph Insights["/api/v1/insights"]
            I1["POST /generate"]
            I2["POST /qa"]
            I3["POST /report"]
        end
    end
```

## n8n Workflow Integration

```mermaid
flowchart LR
    subgraph n8n["⚡ n8n Workflow"]
        Trigger["📅 Schedule Trigger"]
        FetchData["🔗 HTTP: Fetch Data"]
        ValidateN["🔗 HTTP: Validate"]
        TrainN["🔗 HTTP: Train Model"]
        OptimizeN["🔗 HTTP: Optimize"]
        InsightsN["🔗 HTTP: Get Insights"]
        Notify["📧 Send Email/Slack"]
    end

    subgraph API2["FastAPI Server"]
        Endpoints2["REST Endpoints"]
    end

    Trigger --> FetchData
    FetchData --> ValidateN
    ValidateN --> TrainN
    TrainN --> OptimizeN
    OptimizeN --> InsightsN
    InsightsN --> Notify

    FetchData <--> Endpoints2
    ValidateN <--> Endpoints2
    TrainN <--> Endpoints2
    OptimizeN <--> Endpoints2
    InsightsN <--> Endpoints2
```

## Channel Contribution Decomposition

```mermaid
flowchart TB
    subgraph Decomposition["📊 Sales Decomposition"]
        Total["Total Sales: $2.6B"]

        subgraph Base["Base Sales"]
            BaseVal["$1.5B (58%)"]
        end

        subgraph Marketing["Marketing Contribution"]
            DTC_TV["DTC TV: $300M (12%)"]
            DTC_Dig["DTC Digital: $150M (6%)"]
            HCP["HCP Detailing: $400M (15%)"]
            Conf["Conferences: $100M (4%)"]
            Samples["Samples: $120M (5%)"]
            Journal["Journal Ads: $30M (1%)"]
        end

        Total --> Base
        Total --> Marketing
        Marketing --> DTC_TV
        Marketing --> DTC_Dig
        Marketing --> HCP
        Marketing --> Conf
        Marketing --> Samples
        Marketing --> Journal
    end
```

## Budget Optimization Flow

```mermaid
flowchart TB
    subgraph Inputs["📥 Optimization Inputs"]
        TotalBudget["Total Budget: $25M"]
        Current["Current Allocation"]
        Constraints3["Channel Constraints"]
    end

    subgraph Process["⚙️ Optimization Process"]
        ResponseCurves2["Load Response Curves"]
        Objective["Define Objective Function"]
        SLSQP["Run SLSQP Optimizer"]
        Validate3["Validate Constraints"]
    end

    subgraph Outputs["📤 Optimization Outputs"]
        Optimized["Optimized Allocation"]
        Improvement["Expected Improvement: +8.5%"]
        MarginalROI2["Marginal ROI at New Levels"]
    end

    TotalBudget --> ResponseCurves2
    Current --> ResponseCurves2
    Constraints3 --> Validate3

    ResponseCurves2 --> Objective
    Objective --> SLSQP
    SLSQP --> Validate3

    Validate3 --> Optimized
    Optimized --> Improvement
    Optimized --> MarginalROI2
```

## Data Flow Summary

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant API as FastAPI
    participant Svc as Services
    participant DB as Model Storage
    participant Claude as Claude API

    User->>UI: Upload Data
    UI->>Svc: DataService.ingest_data()
    Svc-->>UI: Validation Result

    User->>UI: Configure & Train Model
    UI->>Svc: ModelingService.train_model()
    Svc->>DB: Store Model Artifacts
    Svc-->>UI: Model Results

    User->>UI: View Results
    UI->>Svc: ModelingService.get_decomposition()
    Svc-->>UI: Contribution Data

    User->>UI: Optimize Budget
    UI->>Svc: OptimizationService.optimize_budget()
    Svc->>DB: Load Model
    Svc-->>UI: Optimized Allocation

    User->>UI: Ask Question
    UI->>Svc: InsightService.answer_question()
    Svc->>Claude: Generate Response
    Claude-->>Svc: AI Response
    Svc-->>UI: Answer

    Note over API: n8n can call API endpoints<br/>directly for automation
```

---

## Viewing These Diagrams

These diagrams use [Mermaid](https://mermaid.js.org/) syntax. To view them:

1. **GitHub**: Renders automatically in markdown files
2. **VS Code**: Install "Markdown Preview Mermaid Support" extension
3. **Online**: Paste into [Mermaid Live Editor](https://mermaid.live/)
4. **Streamlit**: Use `streamlit-mermaid` package

## Quick Reference

| Component | Purpose | Key Files |
|-----------|---------|-----------|
| Streamlit UI | Interactive web app | `app.py`, `src/ui/pages/*.py` |
| FastAPI | REST API for n8n | `api.py` |
| DataService | Data ingestion/validation | `src/services/data_service.py` |
| ModelingService | Bayesian MMM training | `src/services/modeling_service.py` |
| OptimizationService | Budget optimization | `src/services/optimization_service.py` |
| InsightService | Claude-powered insights | `src/services/insight_service.py` |
| Agents | Task coordination | `src/agents/*.py` |
