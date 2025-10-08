#!/bin/bash

# Nexus Docker Management Script
# Usage: ./docker-nexus.sh [command]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed and running
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose is not installed. Please install docker-compose first."
        exit 1
    fi
}

# Setup environment
setup_env() {
    print_status "Setting up environment..."

    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from .env.example..."
        cp .env.example .env
        print_warning "Please edit .env file with your configuration before proceeding."
        return 1
    fi

    # Create required directories
    mkdir -p docker/bridge docker/frontend docker/postgres docker/prometheus docker/grafana docker/loki docker/promtail

    print_success "Environment setup complete."
}

# Copy bridge files to docker context
prepare_bridge() {
    print_status "Preparing bridge files for Docker..."

    # Copy requirements.txt to bridge docker context
    cp src/bridge/requirements.txt docker/bridge/
    cp src/bridge/nexus_bridge.py docker/bridge/

    print_success "Bridge files prepared."
}

# Development environment
dev() {
    print_status "Starting development environment..."
    check_docker
    setup_env || return 1
    prepare_bridge

    docker-compose up --build
}

# Production environment
prod() {
    print_status "Starting production environment..."
    check_docker
    setup_env || return 1
    prepare_bridge

    if [ ! -f ".env" ]; then
        print_error "Production requires .env file. Please create it from .env.example"
        exit 1
    fi

    docker-compose -f docker-compose.prod.yml up --build -d

    print_success "Production environment started!"
    print_status "Access your application at: http://localhost"
    print_status "Grafana monitoring: http://localhost:3001"
    print_status "Traefik dashboard: http://localhost:8080"
}

# Stop all services
stop() {
    print_status "Stopping all services..."
    docker-compose down
    docker-compose -f docker-compose.prod.yml down 2>/dev/null || true
    print_success "All services stopped."
}

# Clean up everything
clean() {
    print_warning "This will remove all containers, images, and volumes. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Cleaning up Docker resources..."
        docker-compose down -v --rmi all
        docker-compose -f docker-compose.prod.yml down -v --rmi all 2>/dev/null || true
        docker system prune -f
        print_success "Cleanup complete."
    else
        print_status "Cleanup cancelled."
    fi
}

# Show logs
logs() {
    if [ -z "$2" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f "$2"
    fi
}

# Health check
health() {
    print_status "Checking service health..."

    services=("nexus-bridge" "nexus-frontend" "nexus-redis" "nexus-postgres")

    for service in "${services[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "$service"; then
            status=$(docker inspect --format='{{.State.Health.Status}}' "$service" 2>/dev/null || echo "no healthcheck")
            if [ "$status" = "healthy" ] || [ "$status" = "no healthcheck" ]; then
                print_success "$service: Running"
            else
                print_error "$service: $status"
            fi
        else
            print_error "$service: Not running"
        fi
    done
}

# Backup data
backup() {
    print_status "Creating backup..."
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_dir="backups/backup_$timestamp"

    mkdir -p "$backup_dir"

    # Backup postgres
    docker-compose exec postgres pg_dump -U nexus nexus > "$backup_dir/postgres_backup.sql"

    # Backup volumes
    docker run --rm -v "$(pwd)_bridge_data:/data" -v "$(pwd)/$backup_dir:/backup" alpine tar czf /backup/bridge_data.tar.gz -C /data .
    docker run --rm -v "$(pwd)_redis_data:/data" -v "$(pwd)/$backup_dir:/backup" alpine tar czf /backup/redis_data.tar.gz -C /data .

    print_success "Backup created in $backup_dir"
}

# Restore from backup
restore() {
    if [ -z "$2" ]; then
        print_error "Please specify backup directory: ./docker-nexus.sh restore backups/backup_YYYYMMDD_HHMMSS"
        exit 1
    fi

    backup_dir="$2"

    if [ ! -d "$backup_dir" ]; then
        print_error "Backup directory $backup_dir not found."
        exit 1
    fi

    print_warning "This will restore from $backup_dir. All current data will be replaced. Continue? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Restoring from backup..."

        # Stop services
        docker-compose down

        # Restore postgres
        if [ -f "$backup_dir/postgres_backup.sql" ]; then
            docker-compose up -d postgres
            sleep 10
            docker-compose exec -T postgres psql -U nexus -d nexus < "$backup_dir/postgres_backup.sql"
        fi

        # Restore volumes
        if [ -f "$backup_dir/bridge_data.tar.gz" ]; then
            docker run --rm -v "$(pwd)_bridge_data:/data" -v "$(pwd)/$backup_dir:/backup" alpine tar xzf /backup/bridge_data.tar.gz -C /data
        fi

        if [ -f "$backup_dir/redis_data.tar.gz" ]; then
            docker run --rm -v "$(pwd)_redis_data:/data" -v "$(pwd)/$backup_dir:/backup" alpine tar xzf /backup/redis_data.tar.gz -C /data
        fi

        print_success "Restore complete. Starting services..."
        docker-compose up -d
    else
        print_status "Restore cancelled."
    fi
}

# Update images
update() {
    print_status "Updating Docker images..."
    docker-compose pull
    docker-compose -f docker-compose.prod.yml pull 2>/dev/null || true
    print_success "Images updated."
}

# Show help
help() {
    echo "Nexus Docker Management Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  dev        Start development environment"
    echo "  prod       Start production environment"
    echo "  stop       Stop all services"
    echo "  restart    Restart all services"
    echo "  logs       Show logs (optionally specify service name)"
    echo "  health     Check service health"
    echo "  backup     Create backup of data"
    echo "  restore    Restore from backup"
    echo "  clean      Clean up Docker resources"
    echo "  update     Update Docker images"
    echo "  setup      Setup environment files"
    echo "  help       Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 dev                    # Start development environment"
    echo "  $0 logs nexus-bridge      # Show bridge service logs"
    echo "  $0 restore backups/backup_20231201_120000  # Restore from specific backup"
}

# Main command handler
case "${1:-help}" in
    dev)
        dev
        ;;
    prod)
        prod
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 2
        dev
        ;;
    logs)
        logs "$@"
        ;;
    health)
        health
        ;;
    backup)
        backup
        ;;
    restore)
        restore "$@"
        ;;
    clean)
        clean
        ;;
    update)
        update
        ;;
    setup)
        setup_env
        ;;
    help|*)
        help
        ;;
esac
