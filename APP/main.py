#!/usr/bin/env python3
"""
Simple OneM2M Data Sender with MAPE-K DDoS Protection
======================================================

Sends sample data to the TestGreenhouse1 AE and TestData container.
Includes MAPE-K loop with AI-powered DDoS detection and traffic filtering.
"""

import requests
import time
import uuid
import json
import logging
from datetime import datetime
import random
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import MAPE-K components
from mapek_system import MAPEKSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
BASE_URL = "http://localhost:7579/Mobius"
AE_NAME = "TestGreenhouse1"
CONTAINER_NAME = "TestData"
ORIGINATOR = "admin:admin"
TIMEOUT = 10

# MAPE-K System (Global instance)
mapek_system = MAPEKSystem()

# Initialize MAPE-K system
def initialize_mapek():
    """Initialize MAPE-K components"""
    global mapek_system
    
    logger.info("Initializing MAPE-K DDoS Protection System...")
    
    # Register trusted IoT devices (example)
    mapek_system.register_iot_device("greenhouse_sensor_001", {
        "type": "environmental_sensor",
        "location": "Zone_A1",
        "manufacturer": "IoT_Sensors_Inc"
    })
    
    mapek_system.register_iot_device("greenhouse_actuator_001", {
        "type": "climate_control",
        "location": "Zone_A1", 
        "manufacturer": "SmartClimate_Corp"
    })
    
    # Add local network to whitelist
    mapek_system.add_to_whitelist("127.0.0.1")
    mapek_system.add_to_whitelist("192.168.1.0/24")
    
    logger.info("MAPE-K system initialized successfully")
    logger.info(f"MAPE-K Status: {mapek_system.get_status()}")
    logger.info(f"MAPE-K Stats: {mapek_system.get_stats()}")

# =====================

def print_banner():
    """Print professional banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                IoT Data Sender v3.0 - MAPE-K DDoS Protection                ║
║                          oneM2M Integration Platform                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Target: TestGreenhouse1/TestData                                          ║
║  Protocol: HTTP/oneM2M                                                       ║
║  Data: Augmented Sensor Readings (20+ sensors + metadata)                  ║
║  Protection: AI-powered DDoS detection + MAPE-K loop                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(f"\033[36m{banner}\033[0m")

def generate_sample_data():
    """Generate comprehensive sample sensor data with augmented information"""
    # Core environmental sensors
    sensors = {
        "temperature": round(random.uniform(18.0, 35.0), 2),  # Celsius
        "humidity": round(random.uniform(40.0, 80.0), 2),     # Percentage
        "light": round(random.uniform(100, 1000), 0),         # Lux
        "soil_moisture": round(random.uniform(20.0, 60.0), 2), # Percentage
        "ph": round(random.uniform(6.0, 7.5), 1),             # pH level
        "nutrients": round(random.uniform(100, 500), 0)      # ppm
    }
    
    # Additional augmented sensors
    augmented_sensors = {
        "co2_level": round(random.uniform(400, 1200), 1),      # ppm
        "air_pressure": round(random.uniform(990, 1020), 2),   # hPa
        "wind_speed": round(random.uniform(0, 15), 1),        # km/h
        "uv_index": round(random.uniform(0, 11), 1),          # UV index
        "soil_temperature": round(random.uniform(15, 30), 2),  # Celsius
        "leaf_wetness": round(random.uniform(0, 100), 1),      # Percentage
        "vapor_pressure_deficit": round(random.uniform(0.5, 3.0), 2), # kPa
        "electrical_conductivity": round(random.uniform(0.5, 2.5), 2), # dS/m
        "oxygen_level": round(random.uniform(18, 21), 1),      # Percentage
        "ammonia_level": round(random.uniform(0, 50), 1),      # ppm
        "nitrogen_level": round(random.uniform(100, 300), 0),  # ppm
        "phosphorus_level": round(random.uniform(20, 100), 0), # ppm
        "potassium_level": round(random.uniform(150, 400), 0)  # ppm
    }
    
    # System and device information
    system_info = {
        "device_id": f"GH_{AE_NAME}_{random.randint(1000, 9999)}",
        "firmware_version": f"v{random.randint(1,3)}.{random.randint(0,9)}.{random.randint(0,9)}",
        "battery_level": round(random.uniform(20, 100), 1),     # Percentage
        "signal_strength": round(random.uniform(-80, -30), 0), # dBm
        "cpu_usage": round(random.uniform(10, 80), 1),         # Percentage
        "memory_usage": round(random.uniform(30, 90), 1),      # Percentage
        "uptime_hours": round(random.uniform(1, 720), 1)       # Hours
    }
    
    # Location and spatial data
    location_data = {
        "zone_id": f"Zone_{random.choice(['A', 'B', 'C'])}{random.randint(1,5)}",
        "rack_number": random.randint(1, 20),
        "shelf_level": random.choice(['bottom', 'middle', 'top']),
        "coordinates": {
            "x": round(random.uniform(0, 100), 2),
            "y": round(random.uniform(0, 50), 2),
            "z": round(random.uniform(0, 10), 2)
        }
    }
    
    # Quality and metrics
    quality_metrics = {
        "data_quality_score": round(random.uniform(0.85, 1.0), 3),
        "sensor_health": random.choice(['excellent', 'good', 'fair', 'poor']),
        "calibration_status": random.choice(['calibrated', 'needs_calibration', 'uncalibrated']),
        "last_maintenance": datetime.now().isoformat(),
        "error_count": random.randint(0, 5),
        "warning_count": random.randint(0, 10)
    }
    
    # Combine all data
    data = {
        "sensor_data": {**sensors, **augmented_sensors},
        "system_info": system_info,
        "location_data": location_data,
        "quality_metrics": quality_metrics,
        "timestamp": datetime.now().isoformat(),
        "greenhouse_id": AE_NAME,
        "location": "Test_Lab",
        "status": "active",
        "data_type": "augmented",
        "transmission_id": str(uuid.uuid4()),
        "batch_id": f"Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    return data

def send_data_to_container(ae_name, container_name, data):
    """Send data to oneM2M container with MAPE-K protection"""
    url = f"{BASE_URL}/{ae_name}/{container_name}"
    
    headers = {
        'X-M2M-Origin': ORIGINATOR,
        'X-M2M-RI': str(uuid.uuid4()),
        'Content-Type': 'application/json;ty=4',  # Content Instance
        'Accept': 'application/json'
    }
    
    payload = {
        "m2m:cin": {
            "cnf": "application/json",
            "con": json.dumps(data)
        }
    }
    
    # Create request data for traffic filtering
    request_data = {
        'source_ip': '127.0.0.1',  # Simulate local IP
        'payload': json.dumps(data),
        'user_agent': 'oneM2M-IoT-Client/3.0',
        'device_id': data.get('system_info', {}).get('device_id', 'unknown'),
        'timestamp': datetime.now().isoformat()
    }
    
    # Apply MAPE-K traffic filtering
    filter_result = mapek_system.executor_filter_request(request_data)
    if not filter_result['allowed']:
        logger.warning(f"🚫 Request blocked by MAPE-K filter: {filter_result['reason']}")
        return False
    
    # Generate request ID for monitoring
    request_id = str(uuid.uuid4())
    
    # Monitor request start
    mapek_system.monitor_record_request_start(request_id)
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
        end_time = time.time()
        
        # Monitor request end
        success = response.status_code == 201
        mapek_system.monitor_record_request_end(request_id, success)
        
        if success:
            logger.info(f"✓ Data sent successfully to {ae_name}/{container_name}")
            logger.info(f"  Response time: {(end_time - start_time)*1000:.2f}ms")
            logger.info(f"  Filter: {filter_result['filter_type']}")
            logger.info(f"  Data: Temperature={data['sensor_data']['temperature']}°C, "
                       f"Humidity={data['sensor_data']['humidity']}%")
            
            # Log MAPE-K metrics periodically
            if mapek_system.total_requests % 10 == 0:
                metrics = mapek_system.monitor_collect_metrics()
                logger.info(f"📊 MAPE-K Metrics: RTT={metrics['network_metrics']['rtt_stats']['mean']:.1f}ms, "
                           f"Success Rate={metrics['network_metrics']['success_rate']:.1f}%, "
                           f"CPU={metrics['system_metrics']['cpu_percent']:.1f}%")
            
            return True
        else:
            logger.error(f"✗ Failed to send data - Status: {response.status_code}")
            logger.error(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        mapek_system.monitor_record_request_end(request_id, False)
        logger.error(f"✗ Error sending data: {e}")
        return False

def check_container_exists(ae_name, container_name):
    """Check if container exists"""
    url = f"{BASE_URL}/{ae_name}/{container_name}"
    headers = {
        'X-M2M-Origin': ORIGINATOR,
        'X-M2M-RI': str(uuid.uuid4()),
        'Accept': 'application/json'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def send_single_sample():
    """Send a single sample data point"""
    logger.info("Generating and sending single sample data...")
    
    if not check_container_exists(AE_NAME, CONTAINER_NAME):
        logger.error(f"Container {AE_NAME}/{CONTAINER_NAME} does not exist!")
        logger.error("Please run: python create_resources.py")
        return False
    
    sample_data = generate_sample_data()
    success = send_data_to_container(AE_NAME, CONTAINER_NAME, sample_data)
    
    if success:
        logger.info("✓ Single sample transmission completed successfully!")
    else:
        logger.error("✗ Single sample transmission failed!")
    
    return success

def send_continuous_samples(interval=5, count=10):
    """Send continuous sample data"""
    logger.info(f"Starting continuous data transmission...")
    logger.info(f"  Interval: {interval} seconds")
    logger.info(f"  Count: {count} samples")
    
    if not check_container_exists(AE_NAME, CONTAINER_NAME):
        logger.error(f"Container {AE_NAME}/{CONTAINER_NAME} does not exist!")
        logger.error("Please run: python create_resources.py")
        return False
    
    success_count = 0
    
    for i in range(count):
        logger.info(f"\n--- Sample {i+1}/{count} ---")
        
        sample_data = generate_sample_data()
        if send_data_to_container(AE_NAME, CONTAINER_NAME, sample_data):
            success_count += 1
        
        # Wait between samples (except after the last one)
        if i < count - 1:
            logger.info(f"Waiting {interval} seconds...")
            time.sleep(interval)
    
    logger.info(f"\n✓ Continuous transmission completed!")
    logger.info(f"  Success rate: {success_count}/{count} ({success_count/count*100:.1f}%)")
    
    return success_count == count

def send_custom_data():
    """Send custom data provided by user with augmented information"""
    logger.info("Custom data mode - Enter your data:")
    
    try:
        temp = float(input("Temperature (°C): "))
        humidity = float(input("Humidity (%): "))
        light = float(input("Light (lux): "))
        
        # Generate augmented sensors with custom core values
        augmented_sensors = {
            "co2_level": round(random.uniform(400, 1200), 1),      # ppm
            "air_pressure": round(random.uniform(990, 1020), 2),   # hPa
            "wind_speed": round(random.uniform(0, 15), 1),        # km/h
            "uv_index": round(random.uniform(0, 11), 1),          # UV index
            "soil_temperature": round(random.uniform(15, 30), 2),  # Celsius
            "leaf_wetness": round(random.uniform(0, 100), 1),      # Percentage
            "vapor_pressure_deficit": round(random.uniform(0.5, 3.0), 2), # kPa
            "electrical_conductivity": round(random.uniform(0.5, 2.5), 2), # dS/m
            "oxygen_level": round(random.uniform(18, 21), 1),      # Percentage
            "ammonia_level": round(random.uniform(0, 50), 1),      # ppm
            "nitrogen_level": round(random.uniform(100, 300), 0),  # ppm
            "phosphorus_level": round(random.uniform(20, 100), 0), # ppm
            "potassium_level": round(random.uniform(150, 400), 0), # ppm
            "soil_moisture": round(random.uniform(20.0, 60.0), 2), # Percentage
            "ph": round(random.uniform(6.0, 7.5), 1),             # pH level
            "nutrients": round(random.uniform(100, 500), 0)      # ppm
        }
        
        # System and device information
        system_info = {
            "device_id": f"GH_{AE_NAME}_{random.randint(1000, 9999)}",
            "firmware_version": f"v{random.randint(1,3)}.{random.randint(0,9)}.{random.randint(0,9)}",
            "battery_level": round(random.uniform(20, 100), 1),     # Percentage
            "signal_strength": round(random.uniform(-80, -30), 0), # dBm
            "cpu_usage": round(random.uniform(10, 80), 1),         # Percentage
            "memory_usage": round(random.uniform(30, 90), 1),      # Percentage
            "uptime_hours": round(random.uniform(1, 720), 1)       # Hours
        }
        
        # Location and spatial data
        location_data = {
            "zone_id": f"Zone_{random.choice(['A', 'B', 'C'])}{random.randint(1,5)}",
            "rack_number": random.randint(1, 20),
            "shelf_level": random.choice(['bottom', 'middle', 'top']),
            "coordinates": {
                "x": round(random.uniform(0, 100), 2),
                "y": round(random.uniform(0, 50), 2),
                "z": round(random.uniform(0, 10), 2)
            }
        }
        
        # Quality and metrics
        quality_metrics = {
            "data_quality_score": round(random.uniform(0.85, 1.0), 3),
            "sensor_health": random.choice(['excellent', 'good', 'fair', 'poor']),
            "calibration_status": random.choice(['calibrated', 'needs_calibration', 'uncalibrated']),
            "last_maintenance": datetime.now().isoformat(),
            "error_count": random.randint(0, 5),
            "warning_count": random.randint(0, 10)
        }
        
        custom_data = {
            "sensor_data": {
                "temperature": temp,
                "humidity": humidity,
                "light": light,
                **augmented_sensors
            },
            "system_info": system_info,
            "location_data": location_data,
            "quality_metrics": quality_metrics,
            "timestamp": datetime.now().isoformat(),
            "greenhouse_id": AE_NAME,
            "location": "Test_Lab",
            "status": "active",
            "data_type": "custom_augmented",
            "transmission_id": str(uuid.uuid4()),
            "batch_id": f"Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        success = send_data_to_container(AE_NAME, CONTAINER_NAME, custom_data)
        
        if success:
            logger.info("✓ Custom augmented data sent successfully!")
        else:
            logger.error("✗ Failed to send custom data!")
        
        return success
        
    except ValueError as e:
        logger.error(f"✗ Invalid input: {e}")
        return False

def show_mapek_status():
    """Show current MAPE-K system status"""
    logger.info("\n🔍 MAPE-K System Status:")
    logger.info("=" * 50)
    
    # Monitor status
    metrics = mapek_system.monitor_collect_metrics()
    logger.info(f"📊 Monitor Component:")
    logger.info(f"  Total Requests: {metrics['network_metrics']['total_requests']}")
    logger.info(f"  Success Rate: {metrics['network_metrics']['success_rate']:.1f}%")
    logger.info(f"  Avg RTT: {metrics['network_metrics']['rtt_stats']['mean']:.1f}ms")
    logger.info(f"  Request Rate: {metrics['network_metrics']['request_rate']:.1f} req/s")
    logger.info(f"  CPU Usage: {metrics['system_metrics']['cpu_percent']:.1f}%")
    logger.info(f"  Memory Usage: {metrics['system_metrics']['memory_percent']:.1f}%")
    
    # System status
    system_status = mapek_system.get_status()
    logger.info(f"\n🤖 MAPE-K System:")
    logger.info(f"  Models Trained: {system_status['models_trained']}")
    logger.info(f"  Model 1: {system_status['model1_type']}")
    logger.info(f"  Model 2: {system_status['model2_type']}")
    logger.info(f"  Ensemble Method: {system_status['ensemble_method']}")
    logger.info(f"  DDoS Threshold: {system_status['ddos_threshold']:.3f}")
    logger.info(f"  Blocked IPs: {system_status['blocked_ips_count']}")
    logger.info(f"  Allowed IoT Devices: {system_status['allowed_iot_devices']}")
    logger.info(f"  Requests Analyzed: {system_status['total_requests_analyzed']}")
    
    # Statistics
    stats = mapek_system.get_stats()
    logger.info(f"\n🛡️  Protection Stats:")
    logger.info(f"  Total Requests: {stats['total_requests']}")
    logger.info(f"  Allowed: {stats['allowed_requests']}")
    logger.info(f"  Blocked: {stats['blocked_requests']}")
    logger.info(f"  Rate Limited: {stats['rate_limited_requests']}")
    logger.info(f"  AI Blocked: {stats['ai_blocked_requests']}")
    logger.info(f"  Block Rate: {stats['block_rate']:.1f}%")
    
    # Show blocked IPs
    if mapek_system.blocked_ips:
        logger.info(f"\n🚫 Currently Blocked IPs:")
        for ip, info in mapek_system.blocked_ips.items():
            remaining_time = int(info['expires'] - time.time())
            logger.info(f"  {ip}: {info['reason']} ({remaining_time}s remaining)")

def simulate_ddos_test():
    """Simulate DDoS attack for testing"""
    logger.info("\n🧪 Simulating DDoS Attack Test...")
    
    # Simulate attack from malicious IP
    attack_results = mapek_system.simulate_ddos_attack(
        source_ip="192.168.1.100",
        request_count=50,
        interval=0.05
    )
    
    blocked_count = sum(1 for r in attack_results if not r['allowed'])
    logger.info(f"DDoS Test Results: {blocked_count}/50 requests blocked")
    
    # Show updated status
    show_mapek_status()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='IoT Data Sender with MAPE-K DDoS Protection')
    parser.add_argument('--mode', choices=['single', 'continuous', 'custom', 'status', 'ddos-test'], 
                       default='single', help='Operation mode')
    parser.add_argument('--interval', type=int, default=5, 
                       help='Interval between samples (seconds)')
    parser.add_argument('--count', type=int, default=10, 
                       help='Number of samples to send')
    parser.add_argument('--remote', action='store_true', 
                       help='Use remote server (54.242.158.10)')
    parser.add_argument('--no-mapek', action='store_true',
                       help='Disable MAPE-K protection')
    
    args = parser.parse_args()
    
    # Update BASE_URL if remote mode is requested
    global BASE_URL
    if args.remote:
        BASE_URL = "http://54.242.158.10:7579/Mobius"
        logger.info(f"Using remote server: {BASE_URL}")
    
    print_banner()
    logger.info(f"Target AE: {AE_NAME}")
    logger.info(f"Target Container: {CONTAINER_NAME}")
    logger.info(f"Originator: {ORIGINATOR}")
    logger.info(f"Mode: {args.mode}")
    
    # Initialize MAPE-K unless disabled
    if not args.no_mapek:
        initialize_mapek()
    else:
        logger.warning("⚠️  MAPE-K protection DISABLED")
    
    success = False
    
    if args.mode == 'status':
        show_mapek_status()
        success = True
    elif args.mode == 'ddos-test':
        simulate_ddos_test()
        success = True
    elif args.mode == 'single':
        success = send_single_sample()
    elif args.mode == 'continuous':
        success = send_continuous_samples(args.interval, args.count)
    elif args.mode == 'custom':
        success = send_custom_data()
    
    # Export final metrics
    if not args.no_mapek and args.mode in ['single', 'continuous', 'custom']:
        logger.info("\n📤 Exporting MAPE-K metrics...")
        mapek_system.export_data('mapek_system')
        
        # Show final status
        show_mapek_status()
    
    if success:
        logger.info("\n🎉 All operations completed successfully!")
    else:
        logger.error("\n❌ Some operations failed!")
        exit(1)

if __name__ == "__main__":
    main()
