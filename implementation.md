---
title: Implementation Guide
layout: default
nav_order: 3
description: Complete developer setup and implementation guide
---

> **Author:** Ugwute Charles Ogbonna  
> **Programme:** MSc Software Engineering, University of Bolton  
> **Supervisor:** Aamir Abbas

# Implementation Guide

## System Requirements

### Prerequisites
- **Python 3.10 or higher**
- **PostgreSQL 12+**
- **FFmpeg** for video processing
- **Git** for version control
- **Node.js** (optional, for advanced frontend features)
- **Docker & Docker Compose** (for containerized deployment)

### Hardware Recommendations
- **Development**: 8GB RAM, 4-core CPU
- **Production**: 16GB RAM, 8-core CPU, GPU (optional for ML acceleration)

## Step-by-Step Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Charles9257/LaughterProject.git
cd LaughterProject

# Create and activate virtual environment
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install ffmpeg libpq-dev python3-dev

# Install system dependencies (macOS)
brew install ffmpeg postgresql
```

### 3. Database Configuration

```bash
# Create PostgreSQL database
createdb laughter_db

# Or using PostgreSQL shell
psql -U postgres
CREATE DATABASE laughter_db;
CREATE USER laughter_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE laughter_db TO laughter_user;
```

### 4. Environment Variables

Create a `.env` file in the project root:

```env
# Django Configuration
DJANGO_SECRET_KEY=your-super-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database Configuration
DB_NAME=laughter_db
DB_USER=laughter_user
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432

# Email Configuration (Development)
EMAIL_BACKEND=django.core.mail.backends.console.EmailBackend

# Google OAuth (Optional)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
```

### 5. Database Migration

```bash
# Create and apply migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser
```

### 6. Static Files and Media Setup

```bash
# Collect static files
python manage.py collectstatic

# Create media directories
mkdir -p media/uploads media/analysis_videos
```

### 7. Run Development Server

```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000` to see your application running!

## Project Structure

```
LaughterProject/
├── manage.py
├── requirements.txt
├── .env
├── Dockerfile
├── .github/
│   └── workflows/
│       └── ci.yml
├── LaughterProject/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── laughterApp/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── forms.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── tests.py
│   ├── ml.py
│   ├── migrations/
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   ├── images/
│   │   └── icons/
│   └── templates/
│       ├── base.html
│       ├── index.html
│       ├── system.html
│       ├── admin_dashboard.html
│       └── registration/
├── media/
│   ├── uploads/
│   └── analysis_videos/
└── templates/
```

## Configuration Details

### Settings.py Key Configurations

```python
# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': config('DB_NAME'),
        'USER': config('DB_USER'),
        'PASSWORD': config('DB_PASSWORD'),
        'HOST': config('DB_HOST'),
        'PORT': config('DB_PORT', default='5432'),
    }
}

# Media and Static Files
STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / "laughterApp" / "static"]
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Custom User Model
AUTH_USER_MODEL = 'laughterApp.User'

# Django Allauth Configuration
INSTALLED_APPS = [
    'django.contrib.sites',
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'allauth.socialaccount.providers.google',
    # ... other apps
]

SITE_ID = 1
ACCOUNT_LOGIN_METHODS = {'email', 'username'}
ACCOUNT_SIGNUP_FIELDS = ['username*', 'email*', 'password1*', 'password2*']
```

## Example: Event Schema (JSON)
```json
{
  "timestamp": "2025-08-01T12:00:00Z",
  "session_id": "sess_123",
  "user_id": "anon",
  "laughter": true,
  "emotion": {"happy": 0.82, "neutral": 0.12, "sad": 0.06},
  "latency_ms": 42
}
```

## Troubleshooting Common Issues

### 1. Module Import Errors

```bash
# If you get "ModuleNotFoundError: No module named 'allauth'"
pip install django-allauth

# If you get "ModuleNotFoundError: No module named 'jwt'"
pip install PyJWT

# If you get "ModuleNotFoundError: No module named 'cryptography'"
pip install cryptography
```

### 2. Database Connection Issues

```bash
# Check PostgreSQL service
sudo systemctl status postgresql

# Restart PostgreSQL if needed
sudo systemctl restart postgresql

# Test database connection
psql -U laughter_user -d laughter_db -h localhost
```

### 3. FFmpeg Issues

```bash
# Install FFmpeg on Ubuntu/Debian
sudo apt-get install ffmpeg

# Install FFmpeg on macOS
brew install ffmpeg

# Install FFmpeg on Windows
# Download from https://ffmpeg.org/download.html
```

## Development Best Practices

### 1. Code Quality

```bash
# Install development dependencies
pip install black flake8 isort

# Format code
black .

# Check code quality
flake8 .

# Sort imports
isort .
```

### 2. Testing

```bash
# Run all tests
python manage.py test

# Run specific tests
python manage.py test laughterApp.tests

# Run with coverage
pip install coverage
coverage run --source='.' manage.py test
coverage report
```

### 3. Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and commit
git add .
git commit -m "Add new feature"

# Push to remote
git push origin feature/new-feature
```

## CI/CD Pipeline (Mermaid)
```mermaid
flowchart LR
  A[GitHub Push] --> B[GitHub Actions CI]
  B --> C[Build & Test]
  C --> D[Docker Build Images]
  D --> E[Push to GHCR]
  E --> F[Deploy to AWS EC2]
  F --> G[Health Checks + Rollback]
```

## Production Deployment Guide

### Docker Configuration

#### Production Dockerfile

```dockerfile
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=LaughterProject.settings.production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    python3-dev \
    musl-dev \
    netcat \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

# Run application
CMD ["gunicorn", "LaughterProject.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3"]
```

#### Docker Compose for Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "80:8000"
    environment:
      - DJANGO_SETTINGS_MODULE=LaughterProject.settings.production
    env_file:
      - .env.production
    volumes:
      - /var/log/laughter-analysis:/app/logs
    restart: unless-stopped
    depends_on:
      - redis
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - web
    restart: unless-stopped

volumes:
  redis_data:
```

### AWS EC2 Deployment

#### Launch EC2 Instance

```bash
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --count 1 \
    --instance-type t3.medium \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx \
    --user-data file://user-data.sh
```

#### User Data Script (user-data.sh)

```bash
#!/bin/bash
yum update -y
amazon-linux-extras install docker
service docker start
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Git
yum install git -y

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Configure automatic startup
chkconfig docker on
```

### GitHub Actions CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy to AWS EC2

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: laughter-analysis

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python manage.py test
      env:
        DATABASE_URL: postgres://postgres:postgres@localhost:5432/test_db

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Deploy to EC2
      uses: appleboy/ssh-action@v0.1.5
      with:
        host: ${{ secrets.EC2_HOST }}
        username: ${{ secrets.EC2_USERNAME }}
        key: ${{ secrets.EC2_SSH_KEY }}
        script: |
          cd /home/ec2-user/LaughterProject
          git pull origin main
          docker build -t laughter-app .
          docker stop laughter-container || true
          docker rm laughter-container || true
          docker run -d --name laughter-container -p 80:8000 laughter-app
```

### Environment Variables for Production

Create `.env.production` on your EC2 instance:

```env
# Django Configuration
DJANGO_SECRET_KEY=your-super-secure-production-key
DEBUG=False
ALLOWED_HOSTS=your-domain.com,www.your-domain.com,your-ec2-ip

# Database Configuration
DB_NAME=laughter_analysis_prod
DB_USER=admin
DB_PASSWORD=your-secure-db-password
DB_HOST=your-rds-endpoint.amazonaws.com
DB_PORT=5432

# AWS Configuration
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_STORAGE_BUCKET_NAME=laughter-analysis-static
AWS_S3_REGION_NAME=us-east-1

# Email Configuration
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Security Settings
SECURE_SSL_REDIRECT=True
SECURE_HSTS_SECONDS=31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS=True
SECURE_HSTS_PRELOAD=True
```

## Deployment (AWS EC2 via Docker Compose)
```bash
# on EC2 host
docker login ghcr.io -u <user> -p <token>
docker compose pull
docker compose up -d
docker compose logs -f
```

## Advanced Video Processing & ML Integration

### Real-time Processing Optimization

For production environments, we implement several optimization strategies:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class OptimizedVideoAnalyzer:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        self.detector_cache = {}
    
    def get_detector(self, worker_id):
        """
        Thread-safe detector management
        """
        if worker_id not in self.detector_cache:
            self.detector_cache[worker_id] = FER(mtcnn=True)
        return self.detector_cache[worker_id]
    
    async def analyze_video_async(self, video_path):
        """
        Asynchronous video analysis for better performance
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._analyze_video_sync,
            video_path
        )
        return result
```

### ML Model Performance Monitoring

We implement comprehensive monitoring to track ML model performance:

```python
import time
import logging
from django.core.cache import cache

class MLPerformanceMonitor:
    def __init__(self):
        self.logger = logging.getLogger('ml_performance')
    
    def track_analysis(self, video_path, analysis_result, processing_time):
        """
        Track ML analysis performance metrics
        """
        metrics = {
            'video_size': os.path.getsize(video_path),
            'processing_time': processing_time,
            'confidence': analysis_result.get('confidence', 0),
            'laughter_detected': analysis_result.get('laughter_detected', False),
            'timestamp': time.time()
        }
        
        # Store metrics in cache for real-time monitoring
        cache.set(f"ml_metrics_{int(time.time())}", metrics, timeout=3600)
        
        # Log performance data
        self.logger.info(f"ML Analysis: {metrics}")
        
        return metrics
```

### Integration with Django Views

Here's how the ML components integrate with Django views:

```python
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from .ml_engine import OptimizedVideoAnalyzer, MLPerformanceMonitor

# Initialize global instances
video_analyzer = OptimizedVideoAnalyzer()
performance_monitor = MLPerformanceMonitor()

@login_required
@csrf_exempt
async def analyze_video_endpoint(request):
    """
    Asynchronous video analysis endpoint
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    try:
        # Get uploaded video file
        video_file = request.FILES.get('video')
        if not video_file:
            return JsonResponse({'error': 'No video file provided'}, status=400)
        
        # Save video temporarily
        temp_path = save_temp_video(video_file)
        
        # Convert to optimal format
        converted_path = convert_to_mp4(temp_path)
        
        # Perform ML analysis
        start_time = time.time()
        analysis_result = await video_analyzer.analyze_video_async(converted_path)
        processing_time = time.time() - start_time
        
        # Track performance
        performance_monitor.track_analysis(converted_path, analysis_result, processing_time)
        
        # Clean up temporary files
        cleanup_temp_files([temp_path, converted_path])
        
        return JsonResponse({
            'success': True,
            'analysis': analysis_result,
            'processing_time': round(processing_time, 2)
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
```

## Performance Optimization Strategies

### Database Optimization

Advanced database optimization techniques:

```python
# models.py - Optimized models with indexes and constraints
from django.db import models
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.fields import JSONField

class OptimizedAnalysisResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, db_index=True)
    video = models.FileField(upload_to='videos/', null=True, blank=True)
    
    # Core analysis fields with indexes
    laughter_detected = models.BooleanField(default=False, db_index=True)
    laughter_type = models.CharField(max_length=50, db_index=True)
    confidence = models.FloatField(db_index=True)
    emotion = models.CharField(max_length=50, db_index=True)
    
    # JSON field for detailed results
    detailed_results = JSONField(default=dict)
    
    # Timestamps with indexes
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Processing metadata
    processing_time = models.FloatField(null=True, blank=True)
    model_version = models.CharField(max_length=20, default='v1.0')
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['laughter_detected', 'confidence']),
            models.Index(fields=['emotion', 'laughter_type']),
            GinIndex(fields=['detailed_results']),  # For JSON queries
        ]
```

### Caching Strategy

Comprehensive caching implementation:

```python
from django.core.cache import cache
from django.core.cache.utils import make_template_fragment_key
from django.views.decorators.cache import cache_page
from django.utils.decorators import method_decorator
import hashlib

class CacheManager:
    """
    Centralized cache management
    """
    
    # Cache timeouts (in seconds)
    TIMEOUTS = {
        'user_stats': 300,      # 5 minutes
        'system_stats': 600,    # 10 minutes
        'analytics': 1800,      # 30 minutes
        'ml_results': 3600,     # 1 hour
    }
    
    @classmethod
    def cache_analysis_result(cls, video_file, result):
        """Cache analysis result for identical videos"""
        video_hash = cls._calculate_video_hash(video_file)
        cache_key = cls.get_analysis_cache_key(video_hash)
        cache.set(cache_key, result, cls.TIMEOUTS['ml_results'])
        return cache_key
    
    @classmethod
    def get_cached_analysis(cls, video_file):
        """Get cached analysis result"""
        video_hash = cls._calculate_video_hash(video_file)
        cache_key = cls.get_analysis_cache_key(video_hash)
        return cache.get(cache_key)
    
    @staticmethod
    def _calculate_video_hash(video_file):
        """Calculate hash of video file for caching"""
        hash_md5 = hashlib.md5()
        for chunk in iter(lambda: video_file.read(4096), b""):
            hash_md5.update(chunk)
        video_file.seek(0)  # Reset file pointer
        return hash_md5.hexdigest()
```

## Security Hardening

### Advanced Security Configuration

```python
# security.py - Advanced security middleware and utilities
import hashlib
import hmac
from django.conf import settings
from django.http import HttpResponseForbidden
from django.core.cache import cache
import time

class SecurityMiddleware:
    """
    Advanced security middleware
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Rate limiting
        if not self.check_rate_limit(request):
            return HttpResponseForbidden("Rate limit exceeded")
        
        # Request validation
        if not self.validate_request(request):
            return HttpResponseForbidden("Invalid request")
        
        response = self.get_response(request)
        
        # Add security headers
        self.add_security_headers(response)
        
        return response
    
    def add_security_headers(self, response):
        """Add comprehensive security headers"""
        response['X-Content-Type-Options'] = 'nosniff'
        response['X-Frame-Options'] = 'DENY'
        response['X-XSS-Protection'] = '1; mode=block'
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
        
        # HSTS header for HTTPS
        if settings.SECURE_SSL_REDIRECT:
            response['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

## API Documentation

### Video Analysis Endpoint

```python
@login_required
@csrf_exempt
def api_analyze_video(request):
    """
    API endpoint for video analysis
    
    POST /api/analysis/video/
    Content-Type: multipart/form-data
    
    Parameters:
    - video: File (required)
    - metadata: JSON string (optional)
    
    Returns:
    JSON response with analysis results
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    try:
        video_file = request.FILES.get('video')
        if not video_file:
            return JsonResponse({'error': 'No video file provided'}, status=400)
        
        # Optional metadata
        metadata = {}
        if 'metadata' in request.POST:
            metadata = json.loads(request.POST['metadata'])
        
        # Save video temporarily
        temp_path = save_temporary_video(video_file)
        
        # Process video
        analysis_result = process_video_analysis(temp_path, metadata)
        
        # Save analysis result
        result = AnalysisResult.objects.create(
            user=request.user,
            **analysis_result,
            **metadata
        )
        
        # Clean up
        os.remove(temp_path)
        
        return JsonResponse({
            'success': True,
            'analysis_id': result.id,
            'results': {
                'laughter_detected': result.laughter_detected,
                'laughter_type': result.laughter_type,
                'confidence': result.confidence,
                'emotion': result.emotion
            }
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
```

### Get Analysis Results

```python
@login_required
def api_get_analysis_results(request):
    """
    Get user's analysis results
    
    GET /api/analysis/results/
    
    Query Parameters:
    - limit: Number of results to return (default: 20)
    - offset: Number of results to skip (default: 0)
    - emotion: Filter by emotion type
    - laughter_only: Return only results with laughter detected (true/false)
    """
    try:
        limit = int(request.GET.get('limit', 20))
        offset = int(request.GET.get('offset', 0))
        emotion_filter = request.GET.get('emotion')
        laughter_only = request.GET.get('laughter_only') == 'true'
        
        queryset = AnalysisResult.objects.filter(user=request.user)
        
        if emotion_filter:
            queryset = queryset.filter(emotion__icontains=emotion_filter)
        
        if laughter_only:
            queryset = queryset.filter(laughter_detected=True)
        
        results = queryset.order_by('-created_at')[offset:offset+limit]
        
        data = []
        for result in results:
            data.append({
                'id': result.id,
                'laughter_detected': result.laughter_detected,
                'laughter_type': result.laughter_type,
                'confidence': result.confidence,
                'emotion': result.emotion,
                'created_at': result.created_at.isoformat(),
                'video_url': result.video.url if result.video else None
            })
        
        return JsonResponse({
            'success': True,
            'results': data,
            'total_count': queryset.count()
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
```

### Authentication API

#### User Registration

```python
@csrf_exempt
def api_register(request):
    """
    API endpoint for user registration
    
    POST /api/auth/register/
    Content-Type: application/json
    
    {
        "username": "user123",
        "email": "user@example.com",
        "password": "securepassword",
        "profile": {
            "full_name": "John Doe",
            "age": 25,
            "occupation": "Developer"
        }
    }
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    try:
        data = json.loads(request.body)
        
        # Create user
        user = User.objects.create_user(
            username=data['username'],
            email=data['email'],
            password=data['password']
        )
        
        # Create profile if provided
        if 'profile' in data:
            UserProfile.objects.create(
                user=user,
                **data['profile']
            )
        
        return JsonResponse({
            'success': True,
            'user_id': user.id,
            'message': 'User created successfully'
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)
```

## Admin Dashboard Features

### Dashboard Overview

The admin dashboard provides comprehensive system management capabilities:

1. **System Analytics**: Real-time metrics and performance indicators
2. **User Management**: Complete user lifecycle management
3. **Content Moderation**: Review and manage user submissions
4. **System Monitoring**: Health checks and performance metrics
5. **Data Export**: Comprehensive reporting and data export capabilities

### Results Section

The Results section provides comprehensive analysis data visualization and management:

```python
def get_analysis_statistics():
    """
    Generate comprehensive analysis statistics for dashboard
    """
    results = AnalysisResult.objects.all()
    
    stats = {
        'total_analyses': results.count(),
        'laughter_detected': results.filter(laughter_detected=True).count(),
        'avg_confidence': results.aggregate(Avg('confidence'))['confidence__avg'] or 0,
        'emotion_distribution': get_emotion_distribution(results),
        'daily_analysis_count': get_daily_analysis_count(),
        'top_laughter_types': get_top_laughter_types(results)
    }
    
    return stats

def get_emotion_distribution(results):
    """
    Calculate emotion distribution across all analyses
    """
    emotions = results.values('emotion').annotate(
        count=Count('emotion')
    ).order_by('-count')
    
    return {
        'labels': [e['emotion'] for e in emotions],
        'data': [e['count'] for e in emotions]
    }
```

### User Management

Comprehensive user management with advanced features:

```python
@login_required
@user_passes_test(lambda u: u.is_superuser or hasattr(u, 'adminprofile'))
def manage_users_api(request):
    """
    API endpoint for user management operations
    """
    if request.method == 'GET':
        users = User.objects.annotate(
            analysis_count=Count('analysis_results')
        ).values(
            'id', 'username', 'email', 'date_joined', 
            'is_active', 'analysis_count'
        )
        return JsonResponse({'users': list(users)})
    
    elif request.method == 'POST':
        action = request.POST.get('action')
        user_ids = request.POST.getlist('user_ids')
        
        if action == 'activate':
            User.objects.filter(id__in=user_ids).update(is_active=True)
        elif action == 'deactivate':
            User.objects.filter(id__in=user_ids).update(is_active=False)
        elif action == 'delete':
            User.objects.filter(id__in=user_ids).delete()
        
        return JsonResponse({'status': 'success'})
```

### System Monitoring

Real-time system health monitoring:

```python
import psutil
import redis
from django.db import connection

def get_system_health():
    """
    Comprehensive system health check
    """
    health_data = {
        'timestamp': timezone.now(),
        'database': check_database_health(),
        'redis': check_redis_health(),
        'disk_space': check_disk_space(),
        'memory_usage': check_memory_usage(),
        'cpu_usage': check_cpu_usage(),
        'ml_model_status': check_ml_model_status()
    }
    
    return health_data

def check_database_health():
    """
    Check database connection and performance
    """
    try:
        start_time = time.time()
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        response_time = time.time() - start_time
        
        return {
            'status': 'healthy',
            'response_time': round(response_time * 1000, 2),  # milliseconds
            'active_connections': get_active_db_connections()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }
```

### Reports and Analytics

Advanced reporting capabilities:

```python
class ReportGenerator:
    def __init__(self, start_date=None, end_date=None):
        self.start_date = start_date or timezone.now() - timedelta(days=30)
        self.end_date = end_date or timezone.now()
    
    def generate_comprehensive_report(self):
        """
        Generate comprehensive system report
        """
        report = {
            'report_period': {
                'start': self.start_date,
                'end': self.end_date
            },
            'user_analytics': self.get_user_analytics(),
            'analysis_statistics': self.get_analysis_statistics(),
            'system_performance': self.get_system_performance(),
            'engagement_metrics': self.get_engagement_metrics()
        }
        
        return report
    
    def export_to_csv(self):
        """
        Export report data to CSV format
        """
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Write headers and data
        writer.writerow([
            'Date', 'User Count', 'Analysis Count', 'Laughter Detected',
            'Avg Confidence', 'Top Emotion'
        ])
        
        daily_stats = self.get_daily_statistics()
        for stat in daily_stats:
            writer.writerow([
                stat['date'], stat['user_count'], stat['analysis_count'],
                stat['laughter_detected'], stat['avg_confidence'], stat['top_emotion']
            ])
        
        return output.getvalue()
```


---

© 2025 Ugwute Charles Ogbonna — MSc Software Engineering, University of Bolton.  
Licensed for academic and research use only.
```


---

© 2025 Ugwute Charles Ogbonna — MSc Software Engineering, University of Bolton.  
Licensed for academic and research use only.

