#!/usr/bin/env python3
"""
Simple OneM2M Mobius Resource Creation Script
==============================================

Creates one Application Entity (AE) and one container for testing.
"""

import requests
import uuid
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
BASE_URL = "http://localhost:7579/Mobius"  # Change to "http://54.242.158.10:7579/Mobius" for remote
AE_NAME = "TestGreenhouse1"
ORIGINATOR = "admin:admin"  # Use admin:admin for full permissions
CONTAINER_NAME = "TestData"
TIMEOUT = 10
# =====================

def check_ae_exists(ae_name: str, originator: str) -> bool:
    """
    Check if an Application Entity exists.
    
    Args:
        ae_name (str): The name of the AE to check.
        originator (str): The originator ID to use for the request.
    
    Returns:
        bool: True if AE exists, False otherwise.
    """
    url = f"{BASE_URL}/{ae_name}"
    headers = {
        'X-M2M-Origin': originator,
        'X-M2M-RI': str(uuid.uuid4()),
        'Accept': 'application/json'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def create_ae(ae_name: str, originator: str) -> bool:
    """
    Create an Application Entity (AE) in the OneM2M Mobius server.
    
    Args:
        ae_name (str): The name of the AE to create.
        originator (str): The originator ID for the AE.
    
    Returns:
        bool: True if successful or already exists, False otherwise.
    """
    logger.info(f"Creating AE: {ae_name}")
    
    headers = {
        'X-M2M-Origin': originator,
        'X-M2M-RI': str(uuid.uuid4()),
        'Content-Type': 'application/json;ty=2',
        'Accept': 'application/json'
    }

    payload = {
        "m2m:ae": {
            "rn": ae_name,
            "api": "app-sensor",
            "rr": True
        }
    }

    try:
        response = requests.post(BASE_URL, headers=headers, json=payload, timeout=TIMEOUT)
        
        if response.status_code == 201:
            logger.info(f"✓ Successfully created AE '{ae_name}'")
            return True
        elif response.status_code == 409:
            logger.info(f"✓ AE '{ae_name}' already exists")
            return True
        else:
            logger.error(f"✗ Failed to create AE '{ae_name}' - Status: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Error creating AE '{ae_name}': {e}")
        return False

def create_container(ae_name: str, container_name: str, originator: str, 
                    max_instances: int = 1000) -> bool:
    """
    Create a container under an AE.
    
    Args:
        ae_name (str): The name of the parent AE.
        container_name (str): The name of the container to create.
        originator (str): The originator ID.
        max_instances (int): Maximum number of content instances.
    
    Returns:
        bool: True if successful or already exists, False otherwise.
    """
    logger.info(f"Creating container: {container_name} under AE: {ae_name}")
    
    headers = {
        'X-M2M-Origin': originator,
        'X-M2M-RI': str(uuid.uuid4()),
        'Content-Type': 'application/json;ty=3',
        'Accept': 'application/json'
    }
    
    payload = {
        "m2m:cnt": {
            "rn": container_name,
            "mni": max_instances
        }
    }
    
    url = f"{BASE_URL}/{ae_name}"
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
        
        if response.status_code == 201:
            logger.info(f"✓ Successfully created container '{container_name}' under AE '{ae_name}'")
            return True
        elif response.status_code == 409:
            logger.info(f"✓ Container '{container_name}' already exists under AE '{ae_name}'")
            return True
        else:
            logger.error(f"✗ Failed to create container '{container_name}' - Status: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Error creating container '{container_name}': {e}")
        return False

def main():
    """
    Main function that creates one AE and one container.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple OneM2M Resource Creation')
    parser.add_argument('--remote', action='store_true', help='Use remote server (54.242.158.10)')
    parser.add_argument('--ae-name', default=AE_NAME, help='AE name to create')
    parser.add_argument('--container-name', default=CONTAINER_NAME, help='Container name to create')
    parser.add_argument('--originator', default=ORIGINATOR, help='Originator ID')
    
    args = parser.parse_args()
    
    # Update BASE_URL if remote mode is requested
    global BASE_URL
    if args.remote:
        BASE_URL = "http://54.242.158.10:7579/Mobius"
        logger.info(f"Using remote server: {BASE_URL}")
    
    logger.info("Starting simple resource creation...")
    logger.info(f"AE Name: {args.ae_name}")
    logger.info(f"Container Name: {args.container_name}")
    logger.info(f"Originator: {args.originator}")
    
    # Check if AE exists first
    if not check_ae_exists(args.ae_name, args.originator):
        logger.info(f"AE '{args.ae_name}' does not exist, creating it...")
        ae_success = create_ae(args.ae_name, args.originator)
    else:
        logger.info(f"AE '{args.ae_name}' already exists")
        ae_success = True
    
    # Create container only if AE exists
    if ae_success and check_ae_exists(args.ae_name, args.originator):
        container_success = create_container(args.ae_name, args.container_name, args.originator)
    else:
        logger.error(f"Cannot create container - AE '{args.ae_name}' does not exist")
        container_success = False
    
    if ae_success and container_success:
        logger.info("✓ All operations completed successfully!")
        logger.info(f"Resource structure: /{args.ae_name}/{args.container_name}")
    else:
        logger.error("✗ Some operations failed. Check the logs above for details.")
        exit(1)

if __name__ == "__main__":
    main()