---
title: Methodology
layout: default
nav_order: 4
description: Research methodology, system design, and project management approach
---

> **Author:** Ugwute Charles Ogbonna  
> **Programme:** MSc Software Engineering, University of Bolton  
> **Supervisor:** Aamir Abbas

# Methodology

## Research Philosophy & Design

### Philosophical Approach
- **Pragmatic Research Paradigm:** Combines quantitative performance metrics with qualitative user experience evaluation
- **Mixed-Methods Design:** Utilizes **Concurrent Triangulation** for comprehensive validation
- **Thematic Literature Methodology:** Systematic literature review for technical foundation

### Design Philosophy
```mermaid
flowchart LR
  A[Quantitative<br/>Model Metrics] <-- Triangulate --> B[Qualitative<br/>Usability & Ethics]
  A --> C[Integration & Interpretation]
  B --> C
  C --> D[Actionable Insights<br/>for Iteration]
```

## System Design

### Use Case Diagram

```mermaid
flowchart TD
  subgraph "Laughter Detection Platform"
    subgraph "Users"
      U1[Student]
      U2[Teacher]
      U3[Parent]
      U4[Admin]
    end
    
    subgraph "Core Use Cases"
      UC1[Upload Video]
      UC2[View Analysis Results]
      UC3[Download Reports]
      UC4[Manage User Profiles]
      UC5[Configure Settings]
      UC6[Monitor System Performance]
      UC7[Manage User Accounts]
      UC8[View Analytics Dashboard]
    end
    
    subgraph "System Functions"
      SF1[Process Video]
      SF2[Detect Faces]
      SF3[Analyze Emotions]
      SF4[Classify Laughter]
      SF5[Generate Reports]
      SF6[Store Data]
    end
  end
  
  U1 --> UC1
  U1 --> UC2
  U1 --> UC3
  U2 --> UC1
  U2 --> UC2
  U2 --> UC8
  U3 --> UC2
  U3 --> UC3
  U4 --> UC6
  U4 --> UC7
  U4 --> UC8
  
  UC1 --> SF1
  SF1 --> SF2
  SF2 --> SF3
  SF3 --> SF4
  SF4 --> SF5
  SF5 --> SF6
```

### Activity Diagram: Video Processing Workflow

```mermaid
flowchart TD
  Start([User Uploads Video]) --> Check{Video Format Valid?}
  Check -->|No| Error[Display Format Error]
  Check -->|Yes| Convert[Convert to Standard Format]
  Convert --> Extract[Extract Frames at 30 FPS]
  Extract --> Detect[Face Detection using MTCNN]
  Detect --> Found{Faces Found?}
  Found -->|No| NoFace[Log: No Faces Detected]
  Found -->|Yes| Emotion[Emotion Analysis using FER]
  Emotion --> Classify[Laughter Classification]
  Classify --> Aggregate[Aggregate Results]
  Aggregate --> Store[Store in Database]
  Store --> Generate[Generate Analysis Report]
  Generate --> Notify[Notify User: Processing Complete]
  Notify --> End([End])
  NoFace --> End
  Error --> End
```

### Class Diagram: Core System Components

```mermaid
classDiagram
  class User {
    +int id
    +string username
    +string email
    +string role
    +datetime created_at
    +authenticate()
    +get_profile()
    +update_profile()
  }
  
  class Video {
    +int id
    +string filename
    +string format
    +float duration
    +int user_id
    +datetime uploaded_at
    +validate_format()
    +get_metadata()
  }
  
  class Analysis {
    +int id
    +int video_id
    +json emotion_data
    +json laughter_segments
    +float confidence_score
    +datetime processed_at
    +generate_report()
    +export_results()
  }
  
  class EmotionDetector {
    +string model_path
    +float threshold
    +detect_emotions()
    +preprocess_frame()
    +postprocess_results()
  }
  
  class LaughterClassifier {
    +string model_type
    +dict parameters
    +classify_segment()
    +calculate_confidence()
    +validate_prediction()
  }
  
  class Report {
    +int id
    +int analysis_id
    +string report_type
    +json data
    +datetime generated_at
    +generate_pdf()
    +export_csv()
  }
  
  User ||--o{ Video : uploads
  Video ||--|| Analysis : processes
  Analysis ||--|| Report : generates
  EmotionDetector --> Analysis : performs
  LaughterClassifier --> Analysis : performs
```

### Sequence Diagram: User Authentication & Video Processing

```mermaid
sequenceDiagram
  participant U as User
  participant W as Web Interface
  participant A as Auth Service
  participant V as Video Processor
  participant ML as ML Pipeline
  participant DB as Database
  participant N as Notification Service
  
  U->>W: Login Request
  W->>A: Validate Credentials
  A->>DB: Check User Data
  DB-->>A: User Details
  A-->>W: JWT Token
  W-->>U: Authentication Success
  
  U->>W: Upload Video
  W->>A: Validate Token
  A-->>W: Token Valid
  W->>V: Process Video Request
  V->>DB: Store Video Metadata
  V->>ML: Initialize Processing
  
  ML->>ML: Extract Frames
  ML->>ML: Detect Faces
  ML->>ML: Analyze Emotions
  ML->>ML: Classify Laughter
  ML->>DB: Store Results
  
  DB-->>V: Processing Complete
  V->>N: Send Notification
  N-->>U: Email: Results Ready
  
  U->>W: View Results
  W->>DB: Fetch Analysis
  DB-->>W: Analysis Data
  W-->>U: Display Results
```

## Project Timeline: Gantt Chart (September - December 2024)

```mermaid
gantt
  title Laughter Detection Platform Development Timeline
  dateFormat YYYY-MM-DD
  axisFormat %b %d
  
  section Research & Planning
  Literature Review           :active, lit-review, 2024-09-01, 2024-09-20
  Requirements Analysis       :req-analysis, 2024-09-15, 2024-09-25
  System Architecture Design  :arch-design, 2024-09-20, 2024-09-30
  
  section Development Phase 1
  Django Setup & Configuration :django-setup, 2024-09-25, 2024-10-05
  User Authentication System   :auth-system, 2024-10-01, 2024-10-10
  Database Design & Migration  :db-design, 2024-10-05, 2024-10-12
  Basic UI Development         :basic-ui, 2024-10-08, 2024-10-18
  
  section ML/AI Implementation
  Face Detection Integration   :face-detect, 2024-10-12, 2024-10-22
  Emotion Recognition Model    :emotion-model, 2024-10-18, 2024-10-30
  Laughter Classification     :laughter-class, 2024-10-25, 2024-11-05
  ML Pipeline Optimization    :ml-optimize, 2024-11-01, 2024-11-10
  
  section Testing & Quality Assurance
  Unit Testing with Pytest    :unit-tests, 2024-10-20, 2024-11-15
  API Testing with Postman    :api-tests, 2024-11-05, 2024-11-18
  Performance Testing         :perf-tests, 2024-11-10, 2024-11-20
  Security Testing            :sec-tests, 2024-11-15, 2024-11-25
  
  section Deployment & DevOps
  Docker Containerization     :docker, 2024-11-08, 2024-11-18
  AWS EC2 Setup              :aws-setup, 2024-11-15, 2024-11-22
  CI/CD with GitHub Actions  :cicd, 2024-11-18, 2024-11-25
  Production Deployment      :prod-deploy, 2024-11-22, 2024-11-28
  
  section Documentation & Evaluation
  Technical Documentation    :tech-docs, 2024-11-01, 2024-12-05
  User Testing & Surveys     :user-testing, 2024-11-20, 2024-12-10
  Performance Evaluation     :evaluation, 2024-11-25, 2024-12-15
  Final Report Preparation   :final-report, 2024-12-01, 2024-12-20
```

## Development Methodology

### Agile Framework
- **Kanban Methodology:** Continuous flow approach with visual task management
- **Sprint Duration:** 2-week iterations for rapid development cycles
- **Daily Standups:** Progress tracking and impediment identification
- **Retrospectives:** Continuous improvement and process optimization

### Project Management Tools
- **Trello Boards:** Visual task management with swim lanes:
  - Backlog → In Progress → Testing → Done
  - Priority labeling system (High/Medium/Low)
  - Due date tracking and milestone management
- **GitHub Projects:** Integration with code repository for seamless workflow
- **Slack/Discord:** Team communication and automated notifications

### Software Development Life Cycle (SDLC)

```mermaid
flowchart TD
  A[Requirements Analysis] --> B[System Design]
  B --> C[Implementation]
  C --> D[Testing]
  D --> E[Deployment]
  E --> F[Maintenance]
  F --> A
  
  subgraph "Continuous Integration"
    G[Code Commit]
    H[Automated Testing]
    I[Build & Deploy]
    G --> H --> I
  end
  
  C --> G
```

## Research Methods

### Quantitative Research
- **Performance Metrics Collection:**
  - Accuracy, Precision, Recall, F1-Score
  - Latency measurements (p95 < 100ms target)
  - Throughput analysis (concurrent users)
  - Resource utilization monitoring

- **Statistical Analysis:**
  - Comparative analysis with baseline models
  - Confidence interval calculations
  - Hypothesis testing for performance improvements

### Qualitative Research
- **User Experience Evaluation:**
  - **Likert-Scale Surveys:** 5-point scale for usability assessment
  - **Semi-structured Interviews:** In-depth user feedback collection
  - **Thematic Analysis:** Coding and categorization of user responses

- **Questionnaire Design:**
  - System Usability Scale (SUS) implementation
  - Custom questions for domain-specific evaluation
  - Demographic data collection for stratified analysis

### Sampling Strategy
- **Purposive Sampling:** Target users with relevant domain expertise
- **Stratified Sampling:** Representation across user roles (students, teachers, parents)
- **Sample Size:** Minimum 30 participants per user category for statistical significance

### Mixed-Methods Integration
- **Concurrent Triangulation Design:**
  - Simultaneous quantitative and qualitative data collection
  - Independent analysis followed by integration
  - Convergent findings strengthen conclusions
  - Divergent findings prompt deeper investigation

## Technical Implementation Methods

### Development Tools & Environment
- **IDE:** VS Code (primary), PyCharm (secondary)
- **Version Control:** Git with GitHub repository management
- **Code Quality:** Black formatter, Flake8 linting, pre-commit hooks

### Backend Development
- **Framework:** Django 5.2.4 with Python 3.10+
- **Database:** PostgreSQL (production), SQLite (development/testing)
- **Security Implementation:**
  - JWT (JSON Web Tokens) for authentication
  - CORS configuration for cross-origin requests
  - CSRF protection and secure headers
  - Input validation and sanitization

### Machine Learning Pipeline
- **Frameworks:** TensorFlow 2.x, OpenCV 4.x
- **Model Training:** Jupyter Notebook for experimentation and analysis
- **Face Detection:** MTCNN (Multi-task CNN) implementation
- **Emotion Recognition:** FER (Facial Emotion Recognition) library integration

### Testing Strategy
- **Unit Testing:** Pytest framework with comprehensive test coverage
- **Integration Testing:** Django TestCase for database interactions
- **API Testing:** Postman collections with automated test scripts
- **Load Testing:** Artillery.js for performance validation

### Quality Assurance
- **API Documentation:** Swagger/OpenAPI specifications
- **Manual Testing:** CURL commands for endpoint validation
- **Code Review:** Pull request workflow with mandatory reviews
- **Continuous Monitoring:** Application performance monitoring (APM)

### DevOps & Deployment
- **Containerization:** Docker with multi-stage builds
- **Orchestration:** Kubernetes for scalable deployment
- **Cloud Platform:** AWS EC2 with auto-scaling groups
- **CI/CD Pipeline:** GitHub Actions for automated testing and deployment

### Documentation & Knowledge Sharing
- **Technical Documentation:** Hashnode blog for public sharing
- **Repository Documentation:** GitHub README and wiki pages
- **API Documentation:** Interactive Swagger UI
- **Deployment Guides:** Step-by-step setup instructions

### Analytics & Monitoring
- **Qualitative Analysis:** Monkey Survey for user feedback collection
- **Performance Monitoring:** Custom Django middleware for metrics
- **Error Tracking:** Integrated logging with structured output
- **Business Intelligence:** Custom dashboard for key performance indicators

## Ethical Considerations

### Data Privacy & Security
- **GDPR Compliance:** User consent mechanisms and data portability
- **Data Minimization:** Collection limited to essential functionality
- **Secure Storage:** Encryption at rest and in transit
- **Access Controls:** Role-based permissions and audit trails

### Bias Mitigation
- **Diverse Training Data:** Multi-demographic representation
- **Fairness Testing:** Cross-cultural validation of emotion detection
- **Algorithmic Transparency:** Explainable AI techniques
- **Continuous Monitoring:** Bias detection in production results

### User Consent & Transparency
- **Informed Consent:** Clear explanation of data usage
- **Opt-out Mechanisms:** User control over data processing
- **Regular Communication:** Updates on data handling practices
- **Third-party Audits:** Independent privacy assessments

## Validation & Reliability

### Internal Validity
- **Controlled Environment:** Standardized testing conditions
- **Confounding Variables:** Identification and mitigation strategies
- **Instrumentation:** Calibrated measurement tools and metrics

### External Validity
- **Generalizability:** Multi-domain testing scenarios
- **Population Validity:** Representative user sample selection
- **Ecological Validity:** Real-world usage patterns simulation

### Reliability Measures
- **Test-Retest Reliability:** Consistent results across time periods
- **Inter-rater Reliability:** Agreement between multiple evaluators
- **Internal Consistency:** Cronbach's alpha for survey instruments

### Reproducibility
- **Code Versioning:** Complete source code availability
- **Data Provenance:** Detailed dataset documentation
- **Environment Specification:** Docker containers for consistent deployment
- **Experiment Logging:** Comprehensive MLflow experiment tracking

---

© 2025 Ugwute Charles Ogbonna — MSc Software Engineering, University of Bolton.  
Licensed for academic and research use only.

