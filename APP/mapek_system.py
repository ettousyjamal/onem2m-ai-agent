#!/usr/bin/env python3
"""
MAPE-K DDoS Protection System
=============================

Complete MAPE-K loop implementation with Monitor, Analyzer, Planner, and Executor components
for intelligent DDoS detection and IoT traffic protection.
"""

import time
import json
import logging
import numpy as np
import pickle
import threading
import uuid
import psutil
import statistics
import joblib
import os
from datetime import datetime, timedelta
from collections import deque, defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import ipaddress
import hashlib

logger = logging.getLogger(__name__)

class MAPEKSystem:
    """Complete MAPE-K system for DDoS protection"""
    
    def __init__(self, window_size=1000):
        # Global data structures
        self.window_size = window_size
        self.request_history = deque(maxlen=window_size)
        self.metrics_history = deque(maxlen=window_size)
        self.request_times = deque(maxlen=window_size)
        
        # System state
        self.blocked_ips = {}
        self.allowed_iot_devices = {}
        self.ddos_threshold = 0.7
        self.success_count = 0
        self.total_requests = 0
        
        # ML components - Load pre-trained models from Models folder
        self.scaler = StandardScaler()
        self.scaler_model2 = StandardScaler()
        
        # Model 1: Load Random Forest from Models folder
        model1_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Models', 'best_random_forest_model.joblib')
        try:
            self.model1 = joblib.load(model1_path)
            logger.info(f"Loaded Model 1 (Random Forest) from: {model1_path}")
        except Exception as e:
            logger.error(f"Failed to load Model 1: {e}")
            # Fallback to Isolation Forest
            self.model1 = IsolationForest(contamination=0.1, random_state=42)
            logger.info("Using fallback Model 1: Isolation Forest")
        
        # Model 2: Load XGBoost from Models folder
        model2_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Models', 'best_xgb_model.json')
        try:
            import xgboost as xgb
            self.model2 = xgb.XGBClassifier()
            self.model2.load_model(model2_path)
            logger.info(f"Loaded Model 2 (XGBoost) from: {model2_path}")
        except Exception as e:
            logger.error(f"Failed to load Model 2: {e}")
            # Fallback to Local Outlier Factor
            from sklearn.neighbors import LocalOutlierFactor
            self.model2 = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
            logger.info("Using fallback Model 2: Local Outlier Factor")
        
        self.models_trained = True  # Set to True since we're loading pre-trained models
        
        # Configuration
        self.config = {
            'max_requests_per_minute': 60,
            'max_requests_per_hour': 1000,
            'burst_threshold': 10,
            'block_duration': 300,
            'rate_limit_window': 60,
            'enable_whitelist': True,
            'enable_blacklist': True,
            'enable_ai_filtering': True,
            'enable_rate_limiting': True
        }
        
        # IoT signature patterns
        self.iot_patterns = {
            'user_agents': [
                'Mozilla/5.0 (compatible; IoT-Device/1.0',
                'ESP32-HTTP/1.0',
                'Arduino-Client/1.0',
                'RaspberryPi-IoT/1.0'
            ],
            'payload_patterns': [
                'sensor_data', 'temperature', 'humidity', 
                'greenhouse', 'oneM2M', 'actuator'
            ],
            'request_intervals': (5, 300),
            'payload_sizes': (100, 10000)
        }
        
        # Network tracking
        self.request_counts = defaultdict(lambda: deque(maxlen=1000))
        self.whitelist = set()
        self.blacklist = set()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'allowed_requests': 0,
            'blocked_requests': 0,
            'rate_limited_requests': 0,
            'ai_blocked_requests': 0,
            'whitelisted_requests': 0,
            'start_time': datetime.now().isoformat()
        }
        
        # Knowledge base
        self.knowledge_base = {
            'ddos_patterns': [],
            'iot_baselines': {},
            'threat_intelligence': {},
            'learning_history': [],
            'control_actions': deque(maxlen=100)
        }
        
        # Start background threads
        self.start_background_threads()
    
    # ==================== MONITOR COMPONENT ====================
    
    def monitor_record_request_start(self, request_id):
        """Monitor: Record the start of a request"""
        self.request_times.append({
            'request_id': request_id,
            'start_time': time.time(),
            'timestamp': datetime.now().isoformat()
        })
        
    def monitor_record_request_end(self, request_id, success=True):
        """Monitor: Record the end of a request and calculate RTT"""
        for req in reversed(self.request_times):
            if req['request_id'] == request_id and 'end_time' not in req:
                req['end_time'] = time.time()
                req['rtt'] = (req['end_time'] - req['start_time']) * 1000
                req['success'] = success
                break
        
        self.total_requests += 1
        if success:
            self.success_count += 1
    
    def monitor_get_system_metrics(self):
        """Monitor: Get current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return None
    
    def monitor_calculate_success_rate(self, time_window=60):
        """Monitor: Calculate success rate in the last time_window seconds"""
        cutoff_time = time.time() - time_window
        recent_requests = [req for req in self.request_times 
                          if req.get('start_time', 0) > cutoff_time]
        
        if not recent_requests:
            return 0.0
        
        successful = sum(1 for req in recent_requests if req.get('success', False))
        return (successful / len(recent_requests)) * 100
    
    def monitor_get_rtt_stats(self, time_window=60):
        """Monitor: Calculate RTT statistics"""
        cutoff_time = time.time() - time_window
        recent_rtts = [req['rtt'] for req in self.request_times 
                      if req.get('start_time', 0) > cutoff_time and 'rtt' in req]
        
        if not recent_rtts:
            return {'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'count': 0}
        
        return {
            'mean': statistics.mean(recent_rtts),
            'min': min(recent_rtts),
            'max': max(recent_rtts),
            'std': statistics.stdev(recent_rtts) if len(recent_rtts) > 1 else 0,
            'count': len(recent_rtts)
        }
    
    def monitor_get_request_rate(self, time_window=60):
        """Monitor: Calculate requests per second"""
        cutoff_time = time.time() - time_window
        recent_requests = [req for req in self.request_times 
                          if req.get('start_time', 0) > cutoff_time]
        
        return len(recent_requests) / time_window if time_window > 0 else 0
    
    def monitor_collect_metrics(self):
        """Monitor: Collect all current metrics"""
        system_metrics = self.monitor_get_system_metrics()
        rtt_stats = self.monitor_get_rtt_stats()
        success_rate = self.monitor_calculate_success_rate()
        request_rate = self.monitor_get_request_rate()
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': system_metrics,
            'network_metrics': {
                'rtt_stats': rtt_stats,
                'success_rate': success_rate,
                'request_rate': request_rate,
                'total_requests': self.total_requests,
                'success_count': self.success_count
            }
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    # ==================== ANALYZER COMPONENT ====================
    
    def analyzer_extract_features(self, request_data):
        """Analyzer: Extract features for ML analysis"""
        current_time = time.time()
        
        recent_requests = [req for req in self.request_history 
                          if current_time - time.mktime(
                              datetime.fromisoformat(req['timestamp']).timetuple()) < 60]
        
        features = {
            'requests_per_minute': len(recent_requests),
            'unique_ips': len(set(req['request_data'].get('source_ip', 'unknown') 
                                 for req in recent_requests)),
            'avg_payload_size': np.mean([len(str(req.get('payload', ''))) 
                                        for req in recent_requests]) if recent_requests else 0,
            'payload_variance': np.var([len(str(req.get('payload', ''))) 
                                       for req in recent_requests]) if recent_requests else 0,
            'hour_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'source_ip_hash': hash(request_data.get('source_ip', '')) % 1000,
            'payload_size': len(str(request_data.get('payload', ''))),
            'has_iot_keywords': int(any(keyword in str(request_data.get('payload', '')).lower() 
                                       for keyword in self.iot_patterns['payload_patterns']))
        }
        
        return list(features.values())
    
    def analyzer_detect_anomaly_ml(self, features):
        """Analyzer: ML-based anomaly detection using ensemble of pre-trained models"""
        try:
            # Prepare features for both models
            features_array = np.array(features).reshape(1, -1)
            
            # Model 1 prediction (Random Forest or fallback)
            if hasattr(self.model1, 'predict_proba'):
                # Random Forest - get probability of being anomalous
                model1_proba = self.model1.predict_proba(features_array)[0]
                # Assuming binary classification: [normal, anomalous]
                model1_anomaly = model1_proba[1] if len(model1_proba) > 1 else model1_proba[0]
            elif hasattr(self.model1, 'decision_function'):
                # Isolation Forest fallback
                features_scaled_model1 = self.scaler.transform(features_array)
                model1_score = self.model1.decision_function(features_scaled_model1)[0]
                model1_anomaly = max(0, min(1, (1 - model1_score) / 2))
            else:
                # Basic predict
                model1_pred = self.model1.predict(features_array)[0]
                model1_anomaly = float(model1_pred)
            
            # Model 2 prediction (XGBoost or fallback)
            if hasattr(self.model2, 'predict_proba'):
                # XGBoost - get probability of being anomalous
                model2_proba = self.model2.predict_proba(features_array)[0]
                # Assuming binary classification: [normal, anomalous]
                model2_anomaly = model2_proba[1] if len(model2_proba) > 1 else model2_proba[0]
            elif hasattr(self.model2, 'decision_function'):
                # Local Outlier Factor fallback
                features_scaled_model2 = self.scaler_model2.transform(features_array)
                model2_score = self.model2.decision_function(features_scaled_model2)[0]
                model2_anomaly = max(0, min(1, (1 - model2_score) / 2))
            else:
                # Basic predict
                model2_pred = self.model2.predict(features_array)[0]
                model2_anomaly = float(model2_pred)
            
            # Ensemble prediction: 0.5 * model1 + 0.5 * model2
            ensemble_anomaly_score = 0.5 * model1_anomaly + 0.5 * model2_anomaly
            
            # Ensure score is in [0, 1] range
            ensemble_anomaly_score = max(0.0, min(1.0, ensemble_anomaly_score))
            
            logger.debug(f"ML Ensemble: Model1={model1_anomaly:.3f}, Model2={model2_anomaly:.3f}, Ensemble={ensemble_anomaly_score:.3f}")
            
            return ensemble_anomaly_score
            
        except Exception as e:
            logger.error(f"ML ensemble anomaly detection error: {e}")
            return 0.5
    
    def analyzer_detect_anomaly_rules(self, request_data):
        """Analyzer: Rule-based anomaly detection"""
        score = 0.0
        source_ip = request_data.get('source_ip', 'unknown')
        payload = str(request_data.get('payload', ''))
        
        # High frequency requests
        recent_from_ip = [req for req in self.request_history 
                         if req['request_data'].get('source_ip') == source_ip and
                         time.time() - time.mktime(
                             datetime.fromisoformat(req['timestamp']).timetuple()) < 60]
        
        if len(recent_from_ip) > 30:
            score += 0.3
        
        # Suspicious payload patterns
        if len(payload) > 50000:
            score += 0.2
        
        if not any(keyword in payload.lower() 
                  for keyword in self.iot_patterns['payload_patterns']):
            score += 0.2
        
        # Known malicious patterns
        suspicious_patterns = ['<script>', 'javascript:', 'eval(', 'exec(']
        if any(pattern in payload.lower() for pattern in suspicious_patterns):
            score += 0.3
        
        return min(1.0, score)
    
    def analyzer_match_iot_patterns(self, request_data):
        """Analyzer: Match request against IoT device patterns"""
        confidence = 0.0
        payload = str(request_data.get('payload', ''))
        user_agent = request_data.get('user_agent', '').lower()
        
        # Check IoT user agents
        if any(pattern.lower() in user_agent 
               for pattern in self.iot_patterns['user_agents']):
            confidence += 0.3
        
        # Check IoT payload patterns
        iot_keywords_found = sum(1 for keyword in self.iot_patterns['payload_patterns']
                                if keyword.lower() in payload.lower())
        if iot_keywords_found > 0:
            confidence += 0.4 * (iot_keywords_found / len(self.iot_patterns['payload_patterns']))
        
        # Check payload size
        payload_size = len(payload)
        min_size, max_size = self.iot_patterns['payload_sizes']
        if min_size <= payload_size <= max_size:
            confidence += 0.3
        
        return min(1.0, confidence)
    
    def analyzer_detect_system_anomalies(self):
        """Analyzer: Detect system-level anomalies"""
        current_metrics = self.monitor_collect_metrics()
        anomalies = []
        
        # Check for high request rate
        if current_metrics['network_metrics']['request_rate'] > 100:
            anomalies.append({
                'type': 'high_request_rate',
                'value': current_metrics['network_metrics']['request_rate'],
                'threshold': 100,
                'severity': 'high'
            })
        
        # Check for low success rate
        if current_metrics['network_metrics']['success_rate'] < 50:
            anomalies.append({
                'type': 'low_success_rate',
                'value': current_metrics['network_metrics']['success_rate'],
                'threshold': 50,
                'severity': 'medium'
            })
        
        # Check for high RTT
        if current_metrics['network_metrics']['rtt_stats']['mean'] > 5000:
            anomalies.append({
                'type': 'high_rtt',
                'value': current_metrics['network_metrics']['rtt_stats']['mean'],
                'threshold': 5000,
                'severity': 'medium'
            })
        
        # Check for high CPU usage
        if current_metrics['system_metrics']['cpu_percent'] > 90:
            anomalies.append({
                'type': 'high_cpu',
                'value': current_metrics['system_metrics']['cpu_percent'],
                'threshold': 90,
                'severity': 'high'
            })
        
        return {
            'anomalies': anomalies,
            'metrics': current_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    # ==================== PLANNER COMPONENT ====================
    
    def planner_analyze_request(self, request_data):
        """Planner: Analyze request and make decisions"""
        request_id = str(uuid.uuid4())
        analysis_start = time.time()
        
        # Extract features
        features = self.analyzer_extract_features(request_data)
        
        # ML-based anomaly detection
        anomaly_score = self.analyzer_detect_anomaly_ml(features)
        
        # Rule-based detection
        rule_based_score = self.analyzer_detect_anomaly_rules(request_data)
        
        # IoT pattern matching
        iot_confidence = self.analyzer_match_iot_patterns(request_data)
        
        # Combine scores
        final_score = self._combine_scores(anomaly_score, rule_based_score, iot_confidence)
        
        analysis_result = {
            'request_id': request_id,
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'anomaly_score': anomaly_score,
            'rule_based_score': rule_based_score,
            'iot_confidence': iot_confidence,
            'final_score': final_score,
            'analysis_time': (time.time() - analysis_start) * 1000,
            'request_data': request_data
        }
        
        self.request_history.append(analysis_result)
        return analysis_result
    
    def _combine_scores(self, anomaly_score, rule_based_score, iot_confidence):
        """Planner: Combine different scores into final decision"""
        weights = {
            'anomaly_ml': 0.4,
            'anomaly_rules': 0.3,
            'iot_confidence': 0.3
        }
        
        threat_score = (anomaly_score * weights['anomaly_ml'] + 
                       rule_based_score * weights['anomaly_rules'] - 
                       iot_confidence * weights['iot_confidence'])
        
        return max(0, min(1, threat_score))
    
    def planner_make_decision(self, analysis_result):
        """Planner: Make control decision based on analysis"""
        final_score = analysis_result['final_score']
        source_ip = analysis_result['request_data'].get('source_ip', 'unknown')
        
        decision = {
            'action': 'allow',
            'reason': 'normal_traffic',
            'duration': None,
            'confidence': 1 - final_score,
            'timestamp': datetime.now().isoformat()
        }
        
        if final_score > self.ddos_threshold:
            decision.update({
                'action': 'block',
                'reason': 'ddos_detected',
                'duration': self.config['block_duration'],
                'confidence': final_score
            })
            
            # Add to knowledge base
            self.knowledge_base['ddos_patterns'].append({
                'timestamp': datetime.now().isoformat(),
                'pattern': analysis_result,
                'source_ip': source_ip
            })
        
        elif final_score > 0.5:
            decision.update({
                'action': 'rate_limit',
                'reason': 'suspicious_activity',
                'duration': 60,
                'confidence': final_score
            })
        
        self.knowledge_base['control_actions'].append(decision)
        return decision
    
    # ==================== EXECUTOR COMPONENT ====================
    
    def executor_is_ip_in_list(self, ip, ip_list):
        """Executor: Check if IP is in a list (supports CIDR ranges)"""
        try:
            ip_obj = ipaddress.ip_address(ip)
            
            for entry in ip_list:
                if '/' in entry:
                    network = ipaddress.ip_network(entry, strict=False)
                    if ip_obj in network:
                        return True
                else:
                    if str(ip_obj) == entry:
                        return True
            return False
        except ValueError:
            return False
    
    def executor_check_rate_limit(self, source_ip):
        """Executor: Check if IP has exceeded rate limits"""
        if not self.config['enable_rate_limiting']:
            return True, "Rate limiting disabled"
        
        current_time = time.time()
        window_start = current_time - self.config['rate_limit_window']
        
        ip_requests = self.request_counts[source_ip]
        recent_requests = [req_time for req_time in ip_requests if req_time > window_start]
        
        if len(recent_requests) >= self.config['max_requests_per_minute']:
            return False, f"Rate limit exceeded: {len(recent_requests)}/min"
        
        last_10_seconds = [req_time for req_time in recent_requests 
                          if req_time > current_time - 10]
        if len(last_10_seconds) >= self.config['burst_threshold']:
            return False, f"Burst limit exceeded: {len(last_10_seconds)}/10s"
        
        return True, "Rate limit OK"
    
    def executor_is_blocked(self, source_ip):
        """Executor: Check if IP is currently blocked"""
        if source_ip in self.blocked_ips:
            block_info = self.blocked_ips[source_ip]
            if time.time() < block_info['expires']:
                return True, block_info['reason']
            else:
                del self.blocked_ips[source_ip]
        return False, "Not blocked"
    
    def executor_block_ip(self, source_ip, reason="Manual block", duration=None):
        """Executor: Block an IP address"""
        if duration is None:
            duration = self.config['block_duration']
        
        self.blocked_ips[source_ip] = {
            'reason': reason,
            'blocked_at': time.time(),
            'expires': time.time() + duration,
            'duration': duration
        }
        
        logger.warning(f"IP blocked: {source_ip} - Reason: {reason} - Duration: {duration}s")
    
    def executor_filter_request(self, request_data):
        """Executor: Main filtering function"""
        self.stats['total_requests'] += 1
        
        source_ip = request_data.get('source_ip', 'unknown')
        current_time = time.time()
        
        # Record request timestamp
        self.request_counts[source_ip].append(current_time)
        
        # Check whitelist first
        if self.config['enable_whitelist'] and self.executor_is_ip_in_list(source_ip, self.whitelist):
            self.stats['whitelisted_requests'] += 1
            return self._create_filter_result(True, "Whitelisted IP", "whitelist")
        
        # Check blacklist
        if self.config['enable_blacklist'] and self.executor_is_ip_in_list(source_ip, self.blacklist):
            self.stats['blocked_requests'] += 1
            return self._create_filter_result(False, "Blacklisted IP", "blacklist")
        
        # Check if already blocked
        is_ip_blocked, block_reason = self.executor_is_blocked(source_ip)
        if is_ip_blocked:
            self.stats['blocked_requests'] += 1
            return self._create_filter_result(False, block_reason, "blocked")
        
        # Check rate limits
        rate_ok, rate_reason = self.executor_check_rate_limit(source_ip)
        if not rate_ok:
            self.stats['rate_limited_requests'] += 1
            if "burst limit" in rate_reason:
                self.executor_block_ip(source_ip, f"Auto-block: {rate_reason}", 300)
            return self._create_filter_result(False, rate_reason, "rate_limited")
        
        # AI-based filtering
        if self.config['enable_ai_filtering']:
            analysis = self.planner_analyze_request(request_data)
            decision = self.planner_make_decision(analysis)
            
            if decision['action'] == 'block':
                self.stats['ai_blocked_requests'] += 1
                self.executor_block_ip(source_ip, f"AI DDoS Detection: {decision['reason']}", 600)
                return self._create_filter_result(False, decision['reason'], "ai_blocked")
            elif decision['action'] == 'rate_limit':
                self.stats['rate_limited_requests'] += 1
                return self._create_filter_result(False, decision['reason'], "ai_rate_limited")
        
        # Request passed all checks
        self.stats['allowed_requests'] += 1
        return self._create_filter_result(True, "Request allowed", "allowed")
    
    def _create_filter_result(self, allowed, reason, filter_type):
        """Create standardized filter result"""
        return {
            'allowed': allowed,
            'reason': reason,
            'filter_type': filter_type,
            'timestamp': datetime.now().isoformat(),
            'stats': self.get_stats()
        }
    
    # ==================== KNOWLEDGE & UTILITIES ====================
    
    def register_iot_device(self, device_id, device_info):
        """Register an IoT device as trusted"""
        self.allowed_iot_devices[device_id] = {
            'registration_time': datetime.now().isoformat(),
            'device_info': device_info,
            'baseline_metrics': {}
        }
        logger.info(f"IoT device registered: {device_id}")
    
    def add_to_whitelist(self, ip_or_range):
        """Add IP or IP range to whitelist"""
        try:
            if '/' in ip_or_range:
                network = ipaddress.ip_network(ip_or_range, strict=False)
                self.whitelist.add(str(network))
            else:
                ip = ipaddress.ip_address(ip_or_range)
                self.whitelist.add(str(ip))
            logger.info(f"Added to whitelist: {ip_or_range}")
        except ValueError as e:
            logger.error(f"Invalid IP address/range: {e}")
    
    def get_status(self):
        """Get current system status"""
        # Determine actual model types
        model1_type = "RandomForest" if hasattr(self.model1, 'predict_proba') else "IsolationForest"
        model2_type = "XGBoost" if hasattr(self.model2, 'predict_proba') else "LocalOutlierFactor"
        
        return {
            'models_trained': self.models_trained,
            'model1_type': model1_type,
            'model2_type': model2_type,
            'ensemble_method': '0.5 * model1 + 0.5 * model2',
            'ddos_threshold': self.ddos_threshold,
            'blocked_ips_count': len(self.blocked_ips),
            'allowed_iot_devices': len(self.allowed_iot_devices),
            'total_requests_analyzed': len(self.request_history),
            'recent_control_actions': len(list(self.knowledge_base['control_actions'])[-10:]),
            'config': self.config
        }
    
    def get_stats(self):
        """Get current statistics"""
        current_time = datetime.now()
        start_time = datetime.fromisoformat(self.stats['start_time'])
        uptime = (current_time - start_time).total_seconds()
        
        return {
            **self.stats,
            'uptime_seconds': uptime,
            'blocked_ips_count': len(self.blocked_ips),
            'whitelisted_ips_count': len(self.whitelist),
            'blacklisted_ips_count': len(self.blacklist),
            'requests_per_second': self.stats['total_requests'] / uptime if uptime > 0 else 0,
            'block_rate': (self.stats['blocked_requests'] / self.stats['total_requests'] * 100) 
                         if self.stats['total_requests'] > 0 else 0
        }
    
    def start_background_threads(self):
        """Start background threads for learning and cleanup"""
        def background_learning():
            while True:
                try:
                    time.sleep(60)
                    
                    if len(self.request_history) > 50:
                        self._update_model()
                        self._update_thresholds()
                        
                except Exception as e:
                    logger.error(f"Background learning error: {e}")
        
        def cleanup_expired_blocks():
            while True:
                try:
                    time.sleep(30)
                    
                    current_time = time.time()
                    expired_ips = []
                    
                    for ip, block_info in self.blocked_ips.items():
                        if current_time >= block_info['expires']:
                            expired_ips.append(ip)
                    
                    for ip in expired_ips:
                        del self.blocked_ips[ip]
                        logger.info(f"Block expired for IP: {ip}")
                        
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
        
        learning_thread = threading.Thread(target=background_learning, daemon=True)
        cleanup_thread = threading.Thread(target=cleanup_expired_blocks, daemon=True)
        
        learning_thread.start()
        cleanup_thread.start()
    
    def _update_model(self):
        """Update pre-trained models with new data (incremental learning)"""
        try:
            features = []
            for req in list(self.request_history)[-100:]:
                feature_vector = self.analyzer_extract_features(req['request_data'])
                features.append(feature_vector)
            
            if len(features) > 20:
                features = np.array(features)
                
                # Update Model 1 (Random Forest or fallback)
                if hasattr(self.model1, 'partial_fit'):
                    # For models that support incremental learning
                    try:
                        self.model1.partial_fit(features, np.zeros(len(features)))
                        logger.debug("Model 1 updated with incremental learning")
                    except:
                        logger.debug("Model 1 does not support partial_fit")
                
                # Update Model 2 (XGBoost or fallback)
                if hasattr(self.model2, 'partial_fit'):
                    # For models that support incremental learning
                    try:
                        self.model2.partial_fit(features, np.zeros(len(features)))
                        logger.debug("Model 2 updated with incremental learning")
                    except:
                        logger.debug("Model 2 does not support partial_fit")
                
                # Update scalers if needed
                if len(features) > 50:
                    try:
                        self.scaler.partial_fit(features)
                        self.scaler_model2.partial_fit(features)
                        logger.debug("Scalers updated with new data")
                    except:
                        logger.debug("Scalers do not support partial_fit")
                    
        except Exception as e:
            logger.error(f"Model update error: {e}")
    
    def _update_thresholds(self):
        """Update detection thresholds based on recent patterns"""
        recent_scores = [req['final_score'] for req in list(self.request_history)[-50:]]
        
        if len(recent_scores) > 10:
            avg_score = np.mean(recent_scores)
            std_score = np.std(recent_scores)
            
            new_threshold = min(0.9, max(0.5, avg_score + 2 * std_score))
            self.ddos_threshold = new_threshold
            logger.info(f"Updated DDoS threshold to {new_threshold:.3f}")
    
    def export_data(self, prefix='mapek'):
        """Export all system data"""
        try:
            # Export metrics
            with open(f'{prefix}_metrics.json', 'w') as f:
                json.dump(list(self.metrics_history), f, indent=2)
            
            # Export knowledge base
            export_data = {
                'knowledge_base': dict(self.knowledge_base),
                'status': self.get_status(),
                'stats': self.get_stats(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(f'{prefix}_knowledge.json', 'w') as f:
                json.dump(export_data, f, indent=2)
            
            # Export stats
            with open(f'{prefix}_stats.json', 'w') as f:
                json.dump(self.get_stats(), f, indent=2)
            
            logger.info(f"MAPE-K data exported with prefix: {prefix}")
            
        except Exception as e:
            logger.error(f"Export error: {e}")
    
    def simulate_ddos_attack(self, source_ip, request_count=100, interval=0.1):
        """Simulate a DDoS attack for testing"""
        logger.warning(f"Simulating DDoS attack from {source_ip}: {request_count} requests")
        
        results = []
        for i in range(request_count):
            request_data = {
                'source_ip': source_ip,
                'payload': f'attack_request_{i}',
                'user_agent': 'DDoS-Attack-Tool/1.0',
                'timestamp': datetime.now().isoformat()
            }
            
            result = self.executor_filter_request(request_data)
            results.append(result)
            
            if interval > 0:
                time.sleep(interval)
        
        blocked_count = sum(1 for r in results if not r['allowed'])
        logger.info(f"DDoS simulation complete: {blocked_count}/{request_count} requests blocked")
        
        return results
