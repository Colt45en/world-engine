# 🐳 Docker Migration Guide for Nexus World Engine

## 🚀 **Quick Start (Development)**

```bash
# Windows
docker-nexus.bat dev

# Linux/Mac
./docker-nexus.sh dev
```

Your Nexus system will be available at:
- **Frontend**: http://localhost:3000
- **Bridge API**: http://localhost:8888
- **Health Check**: http://localhost:8888/health

## 📋 **Prerequisites**

### **System Requirements**
- **Docker Desktop** 4.0+ (Windows/Mac) or **Docker Engine** 20.10+ (Linux)
- **docker-compose** 2.0+
- **4GB+ RAM** (8GB recommended for production)
- **2GB+ free disk space**

### **Installation Check**
```bash
# Verify Docker is installed and running
docker --version
docker-compose --version
docker info
```

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Bridge API    │    │   Database      │
│   (React/TS)    │◄──►│   (FastAPI)     │◄──►│   (PostgreSQL)  │
│   Port: 3000    │    │   Port: 8888    │    │   Port: 5432    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │     Redis       │
                    │   (Caching)     │
                    │   Port: 6379    │
                    └─────────────────┘
```

## 🛠️ **Setup Instructions**

### **Step 1: Environment Configuration**
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (Windows: notepad .env, Linux/Mac: nano .env)
# Update these key values:
POSTGRES_PASSWORD=your_secure_password
GRAFANA_PASSWORD=your_grafana_password
DOMAIN=localhost  # Change for production
```

### **Step 2: Choose Environment**

#### **Development (Hot Reload, Debugging)**
```bash
# Windows
docker-nexus.bat dev

# Linux/Mac
chmod +x docker-nexus.sh
./docker-nexus.sh dev
```

#### **Production (Optimized, Monitoring)**
```bash
# Windows
docker-nexus.bat prod

# Linux/Mac
./docker-nexus.sh prod
```

### **Step 3: Verify Deployment**
```bash
# Check service health
docker-nexus.bat health  # Windows
./docker-nexus.sh health # Linux/Mac

# View logs
docker-nexus.bat logs nexus-bridge
```

## 🔧 **Configuration Options**

### **Environment Variables (.env)**
```bash
# Core Configuration
NEXUS_ENV=development|production
MAX_MEMORY_MB=512
MAX_TRAINING_POINTS=1000

# Database
POSTGRES_PASSWORD=secure_password_here
DATABASE_URL=postgresql://nexus:${POSTGRES_PASSWORD}@postgres:5432/nexus

# Redis Cache
REDIS_URL=redis://redis:6379

# Monitoring
GRAFANA_PASSWORD=admin_password

# Production Only
DOMAIN=your-domain.com
ACME_EMAIL=your-email@domain.com
JWT_SECRET=your_32_char_minimum_secret
```

### **Resource Limits**
```yaml
# In docker-compose.yml
deploy:
  resources:
    limits:
      memory: 1G      # Maximum memory
      cpus: '0.5'     # Maximum CPU cores
    reservations:
      memory: 256M    # Reserved memory
      cpus: '0.1'     # Reserved CPU
```

## 📊 **Monitoring & Management**

### **Built-in Monitoring Stack**
- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Health Checks**: http://localhost:8888/health
- **API Stats**: http://localhost:8888/stats

### **Docker Commands**
```bash
# View running containers
docker ps

# Check resource usage
docker stats

# View container logs
docker logs nexus-bridge -f

# Execute commands in container
docker exec -it nexus-bridge bash

# View service details
docker-compose ps
```

### **Management Commands**
```bash
# Start services
docker-nexus.bat dev

# Stop services
docker-nexus.bat stop

# View logs
docker-nexus.bat logs [service-name]

# Health check
docker-nexus.bat health

# Backup data
docker-nexus.bat backup

# Clean everything
docker-nexus.bat clean
```

## 🔐 **Security & Production**

### **Production Security Features**
- **Traefik** reverse proxy with SSL termination
- **Let's Encrypt** automatic SSL certificates
- **Non-root** containers for all services
- **Network isolation** between services
- **Resource limits** to prevent resource exhaustion
- **Health checks** for automatic recovery

### **SSL Setup (Production)**
```bash
# Update .env for production
DOMAIN=your-domain.com
ACME_EMAIL=your-email@domain.com

# Deploy with SSL
./docker-nexus.sh prod
```

### **Firewall Configuration**
```bash
# Allow required ports
# HTTP: 80, HTTPS: 443
# For development also: 3000, 8888

# Linux (ufw)
sudo ufw allow 80
sudo ufw allow 443

# Windows: Configure Windows Defender Firewall
```

## 💾 **Data Management**

### **Backup Strategy**
```bash
# Create backup
docker-nexus.bat backup

# Backup is saved to: backups/backup_YYYYMMDD_HHMMSS/
# Contains:
# - postgres_backup.sql (database)
# - bridge_data.tar.gz (application data)
# - redis_data.tar.gz (cache data)
```

### **Restore Process**
```bash
# Restore from backup
docker-nexus.bat restore backups/backup_20231201_120000

# Manual database restore
docker-compose exec postgres psql -U nexus -d nexus < backup.sql
```

### **Volume Management**
```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect world-engine-feat-v3-1-advanced-math_postgres_data

# Backup volume manually
docker run --rm -v volume_name:/data -v $(pwd):/backup alpine tar czf /backup/backup.tar.gz -C /data .
```

## 🚀 **Scaling & Performance**

### **Horizontal Scaling**
```yaml
# In docker-compose.prod.yml
nexus-bridge:
  deploy:
    replicas: 3  # Run 3 instances
```

### **Load Balancing**
- **Traefik** automatically load balances between replicas
- **Health checks** ensure traffic only goes to healthy instances
- **Sticky sessions** available for stateful operations

### **Performance Optimization**
```bash
# Monitor resource usage
docker stats

# Optimize images
docker system prune -f

# Update to latest images
docker-nexus.bat update

# Check performance metrics
curl http://localhost:8888/stats
```

## 🔧 **Troubleshooting**

### **Common Issues**

#### **"Port already in use"**
```bash
# Find what's using the port
netstat -tulpn | grep :8888  # Linux
netstat -an | findstr :8888  # Windows

# Stop conflicting services
docker-nexus.bat stop
```

#### **"Bridge connection failed"**
```bash
# Check bridge health
curl http://localhost:8888/health

# View bridge logs
docker logs nexus-bridge -f

# Restart bridge
docker-compose restart nexus-bridge
```

#### **"Memory issues"**
```bash
# Check memory usage
docker stats

# Force cleanup
curl -X POST http://localhost:8888/cleanup

# Restart with memory limits
docker-nexus.bat clean
docker-nexus.bat dev
```

#### **"Build failures"**
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache

# Check disk space
df -h  # Linux/Mac
dir   # Windows
```

### **Debug Mode**
```bash
# Enable debug logging
# Add to .env:
DEBUG=true
LOG_LEVEL=debug

# Restart with debug
docker-compose up --build
```

### **Container Shell Access**
```bash
# Access bridge container
docker exec -it nexus-bridge bash

# Access database
docker exec -it nexus-postgres psql -U nexus -d nexus

# Access Redis
docker exec -it nexus-redis redis-cli
```

## 📈 **Monitoring & Alerts**

### **Health Monitoring**
```bash
# Automated health checks
curl http://localhost:8888/health

# Response example:
{
  "status": "healthy",
  "system": "Nexus Core Bridge v1.0",
  "memory_usage": 145.2,
  "training_active": false,
  "uptime_seconds": 3600
}
```

### **Grafana Dashboards**
- **System Overview**: CPU, Memory, Network
- **Application Metrics**: Request rates, Error rates
- **Training Analytics**: Training sessions, Data points
- **Business Metrics**: User engagement, Feature usage

### **Alerting**
```yaml
# Prometheus alerts (docker/prometheus/alerts.yml)
groups:
  - name: nexus
    rules:
      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes > 800000000
        labels:
          severity: warning
      - alert: ServiceDown
        expr: up == 0
        labels:
          severity: critical
```

## 🔄 **CI/CD Integration**

### **GitHub Actions Example**
```yaml
# .github/workflows/deploy.yml
name: Deploy Nexus
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to production
        run: |
          docker-compose -f docker-compose.prod.yml pull
          docker-compose -f docker-compose.prod.yml up -d
```

### **Deployment Strategy**
1. **Blue-Green**: Zero-downtime deployments
2. **Rolling Updates**: Gradual service updates
3. **Health Checks**: Automatic rollback on failure
4. **Backup**: Automated pre-deployment backups

## 🎯 **Benefits of Docker Migration**

### **Before Docker**
- ❌ Manual Python/Node.js setup
- ❌ Environment inconsistencies
- ❌ Complex dependency management
- ❌ No automatic recovery
- ❌ Limited monitoring

### **After Docker**
- ✅ **One-command** deployment
- ✅ **Consistent** environments
- ✅ **Automatic** dependency resolution
- ✅ **Self-healing** with health checks
- ✅ **Comprehensive** monitoring
- ✅ **Easy scaling** and load balancing
- ✅ **Backup/restore** automation
- ✅ **Security** hardening

## 🚀 **Next Steps**

1. **Start Development**: `docker-nexus.bat dev`
2. **Test Features**: Verify chat, training, monitoring
3. **Configure Production**: Update `.env` with your settings
4. **Deploy Production**: `docker-nexus.bat prod`
5. **Setup Monitoring**: Configure Grafana dashboards
6. **Implement Backups**: Schedule regular backups
7. **Scale as Needed**: Add more replicas for load

Your Nexus system is now **production-ready** with enterprise-grade reliability! 🎉
