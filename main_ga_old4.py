import streamlit as st
import autogen
import json
import re
from typing import Dict, List, Optional, Tuple
import logging
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
from io import BytesIO
import base64

# GA4 API imports
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, RunReportRequest, FilterExpression, Filter
from google.oauth2 import service_account
import tempfile

# Load environment variables
load_dotenv()

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ga4_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Enhanced configuration with retry logic
llm_config = {
    "config_list": [{
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": "https://api.openai.com/v1",
        "api_type": "openai"
    }],
    "temperature": 0.1,  # Lower temperature for more consistent results
    "timeout": 120,  # Increased timeout
    "seed": 42,
    "cache_seed": 42,
    "retry_wait_time": 1,
    "max_retry_period": 30
}

# Modern professional color scheme - easy to read and visually appealing
BRAND_COLORS = {
    'primary': '#2563eb',      # Modern blue
    'secondary': '#dc2626',    # Professional red
    'success': '#059669',      # Clean green
    'warning': '#d97706',      # Warm orange
    'danger': '#dc2626',       # Strong red
    'info': '#0891b2',         # Teal blue
    'dark': '#1f2937',         # Dark gray
    'light': '#f9fafb',        # Light gray
    'purple': '#7c3aed',       # Professional purple
    'pink': '#db2777',         # Modern pink
    'indigo': '#4f46e5',       # Rich indigo
    'emerald': '#10b981',      # Emerald green
    'amber': '#f59e0b',        # Amber yellow
    'slate': '#64748b'         # Neutral slate
}

class GA4DataManager:
    """Enhanced GA4 data management with better error handling and validation"""
    
    def __init__(self, property_id: str, service_account_file: str = "service-account.json"):
        self.property_id = property_id
        self.service_account_file = service_account_file
        self.client = None
        self._rate_limit_delay = 1  # Start with 1 second delay
        
    def authenticate(self) -> Dict:
        """Enhanced authentication with better error handling"""
        try:
            if not os.path.exists(self.service_account_file):
                return {
                    "success": False,
                    "error": f"Service account file not found: {self.service_account_file}",
                    "suggestion": "Please ensure the service account JSON file exists in the project directory"
                }
            
            # Validate JSON structure
            with open(self.service_account_file, 'r') as f:
                service_account_info = json.load(f)
            
            # Validate required fields
            required_fields = ["type", "project_id", "private_key", "client_email"]
            missing_fields = [field for field in required_fields if field not in service_account_info]
            
            if missing_fields:
                return {
                    "success": False,
                    "error": f"Missing required fields in service account: {missing_fields}",
                    "suggestion": "Please check your service account JSON file"
                }
            
            # Create credentials
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=['https://www.googleapis.com/auth/analytics.readonly']
            )
            
            self.client = BetaAnalyticsDataClient(credentials=credentials)
            
            # Test connection
            test_request = RunReportRequest(
                property=f"properties/{self.property_id}",
                date_ranges=[DateRange(start_date="yesterday", end_date="today")],
                metrics=[Metric(name="sessions")],
                limit=1
            )
            
            response = self.client.run_report(test_request)
            
            return {
                "success": True,
                "message": f"Successfully authenticated with GA4 property {self.property_id}",
                "test_rows": len(response.rows)
            }
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "suggestion": "Check your service account permissions and property ID"
            }
    
    def get_data_with_retry(self, dimensions: List[str], metrics: List[str], 
                           start_date: str, end_date: str, limit: int = 10000) -> Dict:
        """Enhanced data retrieval with retry logic and rate limiting"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Rate limiting
                if attempt > 0:
                    time.sleep(self._rate_limit_delay * (2 ** attempt))  # Exponential backoff
                
                if not self.client:
                    auth_result = self.authenticate()
                    if not auth_result["success"]:
                        return auth_result
                
                # Validate inputs with enhanced GA4 compatibility checking
                validation_result = self._validate_query_inputs(dimensions, metrics, start_date, end_date)
                if not validation_result["valid"]:
                    return {
                        "success": False,
                        "error": validation_result["error"],
                        "suggestion": validation_result["suggestion"],
                        "compatibility_info": validation_result.get("compatibility_info")
                    }
                
                # Build and execute request
                request = RunReportRequest(
                    property=f"properties/{self.property_id}",
                    date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                    dimensions=[Dimension(name=dim) for dim in dimensions],
                    metrics=[Metric(name=metric) for metric in metrics],
                    limit=min(limit, 100000)  # GA4 API limit
                )
                
                response = self.client.run_report(request)
                
                # Process response
                data = self._process_response(response, dimensions, metrics)
                
                return {
                    "success": True,
                    "data": data,
                    "row_count": len(data),
                    "metadata": {
                        "dimensions": dimensions,
                        "metrics": metrics,
                        "date_range": f"{start_date} to {end_date}",
                        "property_id": self.property_id
                    }
                }
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "error": str(e),
                        "suggestion": "Try reducing the date range or number of dimensions/metrics"
                    }
                
        return {"success": False, "error": "Max retries exceeded"}
    
    def _validate_query_inputs(self, dimensions: List[str], metrics: List[str], 
                              start_date: str, end_date: str) -> Dict:
        """Validate query inputs against GA4 limits and compatibility"""
        
        # Enhanced GA4 compatibility validation
        compatibility_check = RobustQueryIntelligence._validate_ga4_compatibility(dimensions, metrics)
        
        # Check GA4 API limits
        if len(dimensions) > 9:
            return {
                "valid": False,
                "error": f"Too many dimensions ({len(dimensions)}). Maximum is 9.",
                "suggestion": "Reduce the number of dimensions in your query",
                "compatibility_info": compatibility_check
            }
        
        if len(metrics) > 10:
            return {
                "valid": False,
                "error": f"Too many metrics ({len(metrics)}). Maximum is 10.",
                "suggestion": "Reduce the number of metrics in your query",
                "compatibility_info": compatibility_check
            }
        
        # Check for specific GA4 compatibility issues
        if compatibility_check["conflicts"]:
            conflict_details = []
            for conflict in compatibility_check["conflicts"]:
                if conflict["type"] == "metric_conflict":
                    conflicting = ", ".join(conflict["conflicting_metrics"])
                    conflict_details.append(f"Incompatible metrics: {conflicting}")
                elif conflict["type"] == "dimension_metric_conflict":
                    metric = conflict["metric"]
                    dims = ", ".join(conflict["conflicting_dimensions"])
                    conflict_details.append(f"Incompatible combination: {metric} with {dims}")
            
            return {
                "valid": False,
                "error": f"GA4 API compatibility issues found: {'; '.join(conflict_details)}",
                "suggestion": "The system should have auto-resolved these conflicts. If you're seeing this, there may be a bug in the conflict resolution.",
                "compatibility_info": compatibility_check
            }
        
        # Comprehensive GA4 metrics validation
        valid_metrics = {
            # User metrics
            "totalUsers", "newUsers", "activeUsers", "returningUsers",
            # Session metrics  
            "sessions", "sessionsPerUser", "averageSessionDuration",
            # Engagement metrics
            "bounceRate", "engagementRate", "engagedSessions", "userEngagementDuration",
            # Page/Screen metrics
            "screenPageViews", "screenPageViewsPerSession", "screenPageViewsPerUser",
            # Event metrics
            "eventCount", "eventsPerSession", "customEvent",
            # Conversion metrics
            "conversions", "totalRevenue", "purchaseRevenue", "transactions",
            "transactionRevenue", "refundAmount", "shippingAmount", "taxAmount",
            # Ecommerce metrics
            "itemRevenue", "itemsClickedInPromotion", "itemsClickedInList",
            "itemsPurchased", "itemsAddedToCart", "itemsCheckedOut",
            # Publisher metrics (for content sites)
            "publisherAdClicks", "publisherAdImpressions", "adsenseRevenue",
            # Custom metrics (GA4 allows custom metrics)
            "customMetric1", "customMetric2", "customMetric3"
        }
        
        invalid_metrics = [m for m in metrics if m not in valid_metrics]
        if invalid_metrics:
            return {
                "valid": False,
                "error": f"Invalid metrics: {invalid_metrics}",
                "suggestion": f"Use valid metrics: {', '.join(sorted(valid_metrics))}"
            }
        
        # Check date format
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d+daysAgo$',         # NdaysAgo
            r'^(today|yesterday)$'   # today, yesterday
        ]
        
        for date_val in [start_date, end_date]:
            if not any(re.match(pattern, date_val) for pattern in date_patterns):
                return {
                    "valid": False,
                    "error": f"Invalid date format: {date_val}",
                    "suggestion": "Use formats like '2024-01-01', '30daysAgo', 'today', or 'yesterday'"
                }
        
        return {"valid": True}
    
    def _process_response(self, response, dimensions: List[str], metrics: List[str]) -> List[Dict]:
        """Process GA4 API response into structured data"""
        data = []
        
        for row in response.rows:
            row_data = {}
            
            # Process dimensions
            for i, dim in enumerate(dimensions):
                value = row.dimension_values[i].value
                if dim == "date" and len(value) == 8:  # YYYYMMDD format
                    try:
                        row_data[dim] = datetime.strptime(value, '%Y%m%d').strftime('%Y-%m-%d')
                    except:
                        row_data[dim] = value
                elif dim == "pagePath":
                    # Clean up page path for better readability
                    row_data[dim] = self._clean_page_path(value)
                    row_data[dim + "_raw"] = value  # Keep original for reference
                elif dim == "pageTitle":
                    # Clean up page title
                    row_data[dim] = self._clean_page_title(value)
                else:
                    row_data[dim] = value
            
            # Process metrics
            for i, metric in enumerate(metrics):
                value = row.metric_values[i].value
                try:
                    # Convert to appropriate numeric type
                    if '.' in value or 'e' in value.lower():
                        row_data[metric] = float(value)
                    else:
                        row_data[metric] = int(value)
                except (ValueError, TypeError):
                    row_data[metric] = value
            
            data.append(row_data)
        
        return data
    
    def _clean_page_path(self, path: str) -> str:
        """Clean and make page paths more readable"""
        if not path or path.strip() == "":
            return "Unknown Page"
        
        # Handle homepage
        if path == "/" or path == "":
            return "Homepage"
        
        # Remove query parameters for cleaner display
        if "?" in path:
            path = path.split("?")[0]
        
        # Remove trailing slashes
        path = path.rstrip("/")
        
        # If still empty after cleaning, it's homepage
        if not path:
            return "Homepage"
        
        # Limit length for display
        if len(path) > 50:
            return path[:47] + "..."
        
        return path
    
    def _clean_page_title(self, title: str) -> str:
        """Clean and make page titles more readable"""
        if not title or title.strip() == "" or title == "(not set)":
            return "Untitled Page"
        
        # Limit length for display
        if len(title) > 60:
            return title[:57] + "..."
        
        return title

class RobustQueryIntelligence:
    """Robust query parsing system that can handle any user input"""
    
    # Comprehensive GA4 dimensions mapping
    DIMENSION_MAPPING = {
        # Time dimensions
        "time": ["date", "dateHour", "hour", "month", "week", "year"],
        "temporal": ["date", "dateHour", "hour", "month", "week", "year"],
        
        # User dimensions  
        "user": ["userType", "newVsReturning", "cohort", "userAgeBracket", "userGender"],
        "demographic": ["userAgeBracket", "userGender", "userType"],
        
        # Geography dimensions
        "location": ["country", "region", "city", "continent", "subContinent"],
        "geography": ["country", "region", "city", "continent", "subContinent"],
        "geo": ["country", "region", "city", "continent", "subContinent"],
        
        # Technology dimensions
        "device": ["deviceCategory", "deviceModel", "operatingSystem", "browser"],
        "technology": ["deviceCategory", "operatingSystem", "browser", "platform"],
        "tech": ["deviceCategory", "operatingSystem", "browser"],
        
        # Traffic source dimensions
        "source": ["source", "medium", "campaign", "sessionDefaultChannelGroup"],
        "acquisition": ["source", "medium", "campaign", "sessionDefaultChannelGroup", "googleAdsAdGroupName"],
        "marketing": ["campaign", "medium", "source", "sessionDefaultChannelGroup"],
        
        # Content dimensions
        "page": ["pagePath", "pageTitle", "landingPage", "exitPage"],
        "content": ["pagePath", "pageTitle", "landingPage", "exitPage"],
        "website": ["pagePath", "pageTitle", "landingPage", "exitPage"],
        
        # Event dimensions
        "event": ["eventName", "eventCategory", "eventAction", "eventLabel"],
        "interaction": ["eventName", "eventCategory", "eventAction"],
        
        # Ecommerce dimensions
        "product": ["itemId", "itemName", "itemCategory", "itemBrand"],
        "ecommerce": ["transactionId", "affiliation", "itemId", "itemName"],
        "purchase": ["transactionId", "affiliation", "itemId", "itemName"]
    }
    
    # Comprehensive metrics mapping with synonyms
    METRIC_MAPPING = {
        # User metrics
        "users": ["totalUsers", "activeUsers", "newUsers", "returningUsers"],
        "visitors": ["totalUsers", "activeUsers", "newUsers"],
        "people": ["totalUsers", "activeUsers"],
        
        # Session metrics
        "sessions": ["sessions", "sessionsPerUser"],
        "visits": ["sessions", "sessionsPerUser"],
        
        # Engagement metrics
        "engagement": ["engagementRate", "engagedSessions", "userEngagementDuration"],
        "bounce": ["bounceRate"],
        "duration": ["averageSessionDuration", "userEngagementDuration"],
        "time": ["averageSessionDuration", "userEngagementDuration"],
        
        # Page view metrics (avoiding incompatible combinations)
        "pageviews": ["screenPageViews"],
        "views": ["screenPageViews"],
        "pages": ["screenPageViews"],
        "page_efficiency": ["screenPageViewsPerSession"],  # Separate for per-session analysis
        
        # Event metrics
        "events": ["eventCount", "eventsPerSession"],
        "interactions": ["eventCount", "eventsPerSession"],
        
        # Conversion metrics
        "conversions": ["conversions"],
        "goals": ["conversions"],
        
        # Revenue metrics
        "revenue": ["totalRevenue", "purchaseRevenue", "transactionRevenue"],
        "sales": ["totalRevenue", "purchaseRevenue", "transactions"],
        "money": ["totalRevenue", "purchaseRevenue"],
        
        # Ecommerce metrics
        "transactions": ["transactions"],
        "purchases": ["transactions", "itemsPurchased"],
        "cart": ["itemsAddedToCart"],
        "checkout": ["itemsCheckedOut"]
    }
    
    # GA4 API Compatibility Rules
    INCOMPATIBLE_METRIC_COMBINATIONS = {
        # Page view incompatibilities  
        frozenset(["screenPageViews", "screenPageViewsPerSession"]),
        frozenset(["screenPageViews", "screenPageViewsPerUser"]), 
        frozenset(["screenPageViewsPerSession", "screenPageViewsPerUser"]),
        
        # Event incompatibilities
        frozenset(["eventCount", "eventsPerSession"]),
        
        # User metric incompatibilities (some combinations don't work well)
        frozenset(["newUsers", "returningUsers"]),  # Usually better to use one or the other
        
        # Revenue metric incompatibilities
        frozenset(["totalRevenue", "purchaseRevenue", "transactionRevenue"]),  # Can cause double counting
    }
    
    # GA4 Dimension-Metric Incompatibilities 
    INCOMPATIBLE_DIMENSION_METRIC_COMBINATIONS = {
        # User-scoped dimensions cannot be combined with page-scoped metrics
        "screenPageViews": {
            "incompatible_dimensions": [
                # User-scoped dimensions (definitely incompatible)
                "userType", "newVsReturning", "userAgeBracket", "userGender", 
                "cohort", "userFirstTouchTimestamp"
            ],
            "reason": "User-scoped dimensions are incompatible with page-scoped metrics like screenPageViews"
        },
        "screenPageViewsPerSession": {
            "incompatible_dimensions": [
                "userType", "newVsReturning", "userAgeBracket", "userGender",
                "cohort"
            ],
            "reason": "User-scoped dimensions are incompatible with session-scoped page metrics"
        },
        "screenPageViewsPerUser": {
            "incompatible_dimensions": [
                "pagePath", "pageTitle", "landingPage", "exitPage",
                "eventName", "eventCategory", "eventAction"
            ],
            "reason": "Page-scoped dimensions are incompatible with user-scoped page metrics"
        },
        
        # Event metrics with incompatible dimensions
        "eventCount": {
            "incompatible_dimensions": [
                "userType", "newVsReturning", "userAgeBracket", "userGender"
            ],
            "reason": "User-scoped dimensions often incompatible with event-scoped metrics"
        },
        
        # Revenue metrics with certain dimension restrictions
        "totalRevenue": {
            "incompatible_dimensions": [
                "pagePath", "pageTitle"  # Page-level dimensions with transaction-level metrics
            ],
            "reason": "Page-scoped dimensions are incompatible with transaction-scoped metrics"
        },
        "purchaseRevenue": {
            "incompatible_dimensions": [
                "pagePath", "pageTitle"
            ],
            "reason": "Page-scoped dimensions are incompatible with purchase-scoped metrics"
        }
    }
    
    # Preferred metrics when conflicts arise
    METRIC_PREFERENCES = {
        "screenPageViews": 1.0,  # Prefer basic page views
        "screenPageViewsPerSession": 0.8,
        "screenPageViewsPerUser": 0.6,
        
        "sessions": 1.0,  # Prefer sessions over user metrics for most analysis
        "totalUsers": 0.9,
        "activeUsers": 0.8,
        "newUsers": 0.7,
        "returningUsers": 0.7,
        
        "totalRevenue": 1.0,  # Prefer total revenue
        "purchaseRevenue": 0.8,
        "transactionRevenue": 0.6,
        
        "eventCount": 1.0,  # Prefer total event count
        "eventsPerSession": 0.8,
    }

    # Intent patterns with flexible matching
    INTENT_PATTERNS = {
        "traffic_analysis": {
            "keywords": ["traffic", "visitors", "sessions", "users", "visits", "volume", "audience"],
            "priority": 1.0,
            "default_dimensions": ["date"],
            "default_metrics": ["sessions", "totalUsers", "screenPageViews"]
        },
        "page_performance": {
            "keywords": ["pages", "content", "popular", "top", "best", "performing", "page views"],
            "priority": 1.2,
            "default_dimensions": ["pagePath", "pageTitle"],
            "default_metrics": ["screenPageViews", "sessions"]
        },
        "source_analysis": {
            "keywords": ["source", "channel", "referral", "campaign", "acquisition", "where", "how found"],
            "priority": 1.1,
            "default_dimensions": ["source", "medium", "sessionDefaultChannelGroup"],
            "default_metrics": ["sessions", "totalUsers"]
        },
        "device_analysis": {
            "keywords": ["device", "mobile", "desktop", "tablet", "platform", "browser", "technology"],
            "priority": 1.0,
            "default_dimensions": ["deviceCategory", "operatingSystem"],
            "default_metrics": ["sessions", "totalUsers"]
        },
        "geographic_analysis": {
            "keywords": ["country", "location", "geographic", "region", "city", "where"],
            "priority": 1.0,
            "default_dimensions": ["country", "city"],
            "default_metrics": ["sessions", "totalUsers"]
        },
        "engagement_analysis": {
            "keywords": ["engagement", "bounce", "duration", "time", "interaction", "quality"],
            "priority": 1.1,
            "default_dimensions": ["date"],
            "default_metrics": ["engagementRate", "bounceRate", "averageSessionDuration"]
        },
        "conversion_analysis": {
            "keywords": ["conversion", "goal", "revenue", "sales", "purchase", "transaction", "money"],
            "priority": 1.3,
            "default_dimensions": ["date"],
            "default_metrics": ["conversions", "totalRevenue", "transactions"]
        },
        "ecommerce_analysis": {
            "keywords": ["product", "item", "cart", "checkout", "purchase", "ecommerce", "shop"],
            "priority": 1.2,
            "default_dimensions": ["itemName", "itemCategory"],
            "default_metrics": ["itemsPurchased", "itemRevenue", "itemsAddedToCart"]
        }
    }
    
    @classmethod
    def parse_user_request(cls, user_request: str) -> Dict:
        """Advanced natural language query parsing with robust fallbacks"""
        request_lower = user_request.lower().strip()
        
        if not request_lower:
            return cls._get_default_query("Empty query provided")
        
        # Detect intents with scoring
        intent_scores = cls._detect_intents(request_lower)
        
        # Extract dimensions and metrics from query
        detected_dimensions = cls._extract_dimensions(request_lower)
        detected_metrics = cls._extract_metrics(request_lower)
        
        # Get primary intent
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else "traffic_analysis"
        confidence = intent_scores.get(primary_intent, 0.3)
        
        # Build final query configuration 
        # Check if this is a comprehensive query (multiple intents with reasonable scores)
        high_scoring_intents = [intent for intent, score in intent_scores.items() if score > 0.8]
        is_comprehensive = len(high_scoring_intents) > 2
        
        # Check if this is a specific device comparison query that needs pure intent defaults
        is_device_comparison = (primary_intent == "device_analysis" and 
                              confidence > 1.5 and 
                              any(word in request_lower for word in ["mobile", "desktop", "compare", "vs"]))
        
        if is_device_comparison:
            # Device comparison queries - use ONLY intent defaults to avoid dimension conflicts
            intent_defaults = cls.INTENT_PATTERNS[primary_intent]
            final_dimensions = intent_defaults["default_dimensions"]
            final_metrics = intent_defaults["default_metrics"]
        elif is_comprehensive:
            # Comprehensive queries - preserve original behavior (mixed dimensions/metrics)
            final_dimensions = detected_dimensions if detected_dimensions else cls.INTENT_PATTERNS[primary_intent]["default_dimensions"]
            final_metrics = detected_metrics if detected_metrics else cls.INTENT_PATTERNS[primary_intent]["default_metrics"]
        else:
            # Standard queries - use detected with intent defaults as backup
            final_dimensions = detected_dimensions if detected_dimensions else cls.INTENT_PATTERNS[primary_intent]["default_dimensions"]
            final_metrics = detected_metrics if detected_metrics else cls.INTENT_PATTERNS[primary_intent]["default_metrics"]
        
        # Validate GA4 compatibility and resolve conflicts
        original_metrics = final_metrics.copy()
        original_dimensions = final_dimensions.copy()
        
        # First resolve metric-metric conflicts
        final_metrics = cls._resolve_metric_conflicts(final_metrics)
        
        # EMERGENCY FIX: If screenPageViews is present with any non-page dimensions, remove it
        if "screenPageViews" in final_metrics:
            safe_page_dimensions = ["pagePath", "pageTitle", "landingPage", "exitPage", "date"]
            has_non_page_dimensions = any(dim not in safe_page_dimensions for dim in final_dimensions)
            
            if has_non_page_dimensions:
                logger.warning(f"EMERGENCY - Removing screenPageViews due to potentially incompatible dimensions: {final_dimensions}")
                final_metrics.remove("screenPageViews")
                
                # Add safer alternatives
                if "sessions" not in final_metrics:
                    final_metrics.append("sessions")
                if "totalUsers" not in final_metrics:
                    final_metrics.append("totalUsers")
        
        # Then resolve dimension-metric conflicts
        logger.info(f"DEBUG: Before dimension-metric resolution - Dimensions: {final_dimensions}, Metrics: {final_metrics}")
        final_dimensions, final_metrics = cls._resolve_dimension_metric_conflicts(final_dimensions, final_metrics)
        logger.info(f"DEBUG: After dimension-metric resolution - Dimensions: {final_dimensions}, Metrics: {final_metrics}")
        
        # Track if conflicts were auto-resolved
        metric_conflicts_resolved = len(original_metrics) != len(final_metrics) or set(original_metrics) != set(final_metrics)
        dimension_conflicts_resolved = len(original_dimensions) != len(final_dimensions) or set(original_dimensions) != set(final_dimensions)
        conflicts_resolved = metric_conflicts_resolved or dimension_conflicts_resolved
        
        # Validate and limit dimensions/metrics for GA4 API limits
        final_dimensions = final_dimensions[:9]  # GA4 limit
        final_metrics = final_metrics[:10]  # GA4 limit
        
        # Generate interpretation with conflict resolution info
        interpretation = cls._generate_interpretation(user_request, primary_intent, final_dimensions, final_metrics)
        if conflicts_resolved:
            interpretation += " | ⚡ Auto-resolved GA4 compatibility conflicts"
        
        # Add emergency fix notification
        emergency_fix_applied = "screenPageViews" in original_metrics and "screenPageViews" not in final_metrics
        if emergency_fix_applied:
            interpretation += " | 🚨 Applied emergency screenPageViews compatibility fix"
        
        return {
            "dimensions": final_dimensions,
            "metrics": final_metrics,
            "type": primary_intent.replace("_analysis", ""),
            "confidence": confidence,
            "detected_intents": intent_scores,
            "query_interpretation": interpretation,
            "conflicts_resolved": conflicts_resolved,
            "original_metrics": original_metrics if metric_conflicts_resolved else None,
            "original_dimensions": original_dimensions if dimension_conflicts_resolved else None,
            "metric_conflicts_resolved": metric_conflicts_resolved,
            "dimension_conflicts_resolved": dimension_conflicts_resolved,
            "emergency_fix_applied": emergency_fix_applied
        }
    
    @classmethod
    def _detect_intents(cls, query: str) -> Dict[str, float]:
        """Detect multiple intents in a query with confidence scores"""
        intent_scores = {}
        
        for intent, config in cls.INTENT_PATTERNS.items():
            score = 0.0
            matched_keywords = []
            
            for keyword in config["keywords"]:
                if keyword in query:
                    score += config["priority"]
                    matched_keywords.append(keyword)
            
            # Bonus for multiple keyword matches
            if len(matched_keywords) > 1:
                score *= 1.2
            
            # Bonus for exact phrase matches
            for keyword in matched_keywords:
                if f" {keyword} " in f" {query} ":
                    score *= 1.1
            
            if score > 0:
                intent_scores[intent] = score
        
        return intent_scores
    
    @classmethod
    def _extract_dimensions(cls, query: str) -> List[str]:
        """Extract relevant dimensions from the query"""
        dimensions = []
        
        for category, dim_list in cls.DIMENSION_MAPPING.items():
            if category in query:
                dimensions.extend(dim_list[:2])  # Limit to avoid overload
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(dimensions))
    
    @classmethod
    def _extract_metrics(cls, query: str) -> List[str]:
        """Extract relevant metrics from the query"""
        metrics = []
        
        for category, metric_list in cls.METRIC_MAPPING.items():
            if category in query:
                metrics.extend(metric_list[:3])  # Limit to avoid overload
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(metrics))
    
    @classmethod
    def _generate_interpretation(cls, original_query: str, intent: str, dimensions: List[str], metrics: List[str]) -> str:
        """Generate human-readable interpretation of how the query was understood"""
        intent_readable = intent.replace("_", " ").title()
        
        interpretation = f"Interpreted as: {intent_readable}"
        
        if dimensions:
            interpretation += f" | Analyzing by: {', '.join(dimensions)}"
        
        if metrics:
            interpretation += f" | Measuring: {', '.join(metrics)}"
        
        return interpretation
    
    @classmethod
    def _resolve_metric_conflicts(cls, metrics: List[str]) -> List[str]:
        """Resolve GA4 API metric compatibility conflicts"""
        if not metrics:
            return metrics
            
        # Check for incompatible combinations
        metrics_set = set(metrics)
        resolved_metrics = list(metrics)  # Start with original list
        
        for incompatible_set in cls.INCOMPATIBLE_METRIC_COMBINATIONS:
            # Check if any incompatible combination exists in our metrics
            intersection = metrics_set.intersection(incompatible_set)
            if len(intersection) > 1:  # Conflict found
                # Keep only the highest priority metric from the conflicting set
                conflicting_metrics = list(intersection)
                
                # Sort by preference (higher preference = better)
                conflicting_metrics.sort(
                    key=lambda m: cls.METRIC_PREFERENCES.get(m, 0.5), 
                    reverse=True
                )
                
                # Keep the highest priority metric, remove others
                preferred_metric = conflicting_metrics[0]
                metrics_to_remove = conflicting_metrics[1:]
                
                for metric_to_remove in metrics_to_remove:
                    if metric_to_remove in resolved_metrics:
                        resolved_metrics.remove(metric_to_remove)
        
        return resolved_metrics
    
    @classmethod
    def _resolve_dimension_metric_conflicts(cls, dimensions: List[str], metrics: List[str]) -> Tuple[List[str], List[str]]:
        """Resolve GA4 API dimension-metric compatibility conflicts"""
        if not dimensions or not metrics:
            return dimensions, metrics
        
        resolved_dimensions = list(dimensions)
        resolved_metrics = list(metrics)
        
        # Check each metric for incompatible dimensions
        for metric in metrics:
            if metric in cls.INCOMPATIBLE_DIMENSION_METRIC_COMBINATIONS:
                incompatible_dims = cls.INCOMPATIBLE_DIMENSION_METRIC_COMBINATIONS[metric]["incompatible_dimensions"]
                
                # Find conflicting dimensions
                conflicting_dimensions = [dim for dim in resolved_dimensions if dim in incompatible_dims]
                
                if conflicting_dimensions:
                    # Decide whether to remove the metric or the dimensions
                    # Priority: Keep more important analysis components
                    
                    # If it's a core page metric and we have page dimensions, keep the combination
                    if metric == "screenPageViews" and any(dim in ["pagePath", "pageTitle"] for dim in resolved_dimensions):
                        # Remove conflicting user dimensions, keep page dimensions and metric
                        for dim in conflicting_dimensions:
                            if dim in resolved_dimensions:
                                resolved_dimensions.remove(dim)
                    
                    # If it's revenue metrics with page dimensions, remove the page dimensions
                    elif metric in ["totalRevenue", "purchaseRevenue"] and any(dim in ["pagePath", "pageTitle"] for dim in conflicting_dimensions):
                        for dim in conflicting_dimensions:
                            if dim in resolved_dimensions:
                                resolved_dimensions.remove(dim)
                    
                    # For other cases, try to substitute the metric with a safer alternative
                    else:
                        if metric in resolved_metrics:
                            resolved_metrics.remove(metric)
                            
                            # Add safer alternative metrics
                            if metric == "screenPageViews":
                                # Replace with session-based metrics that are more compatible
                                if "sessions" not in resolved_metrics:
                                    resolved_metrics.append("sessions")
                                if "totalUsers" not in resolved_metrics:
                                    resolved_metrics.append("totalUsers")
                            
                            logger.warning(f"Removed conflicting metric '{metric}' due to dimensions: {conflicting_dimensions}")
        
        # If we removed all metrics, add safe fallback metrics
        if not resolved_metrics:
            resolved_metrics = ["sessions", "totalUsers"]
        
        # If we removed all dimensions, add safe fallback dimension
        if not resolved_dimensions:
            resolved_dimensions = ["date"]
        
        return resolved_dimensions, resolved_metrics
    
    @classmethod
    def _validate_ga4_compatibility(cls, dimensions: List[str], metrics: List[str]) -> Dict:
        """Validate GA4 API compatibility and provide suggestions"""
        
        # Check for incompatible metric combinations
        metrics_set = set(metrics)
        conflicts = []
        
        for incompatible_set in cls.INCOMPATIBLE_METRIC_COMBINATIONS:
            intersection = metrics_set.intersection(incompatible_set)
            if len(intersection) > 1:
                conflicts.append({
                    "type": "metric_conflict",
                    "conflicting_metrics": list(intersection),
                    "suggestion": f"Cannot use {' and '.join(intersection)} together. Consider using only one."
                })
        
        # Check for dimension-metric incompatibilities
        for metric in metrics:
            if metric in cls.INCOMPATIBLE_DIMENSION_METRIC_COMBINATIONS:
                incompatible_dims = cls.INCOMPATIBLE_DIMENSION_METRIC_COMBINATIONS[metric]["incompatible_dimensions"]
                conflicting_dimensions = [dim for dim in dimensions if dim in incompatible_dims]
                
                if conflicting_dimensions:
                    reason = cls.INCOMPATIBLE_DIMENSION_METRIC_COMBINATIONS[metric]["reason"]
                    conflicts.append({
                        "type": "dimension_metric_conflict",
                        "metric": metric,
                        "conflicting_dimensions": conflicting_dimensions,
                        "reason": reason,
                        "suggestion": f"Cannot use metric '{metric}' with dimensions: {', '.join(conflicting_dimensions)}. {reason}"
                    })
        
        # Check dimension/metric limits
        issues = []
        if len(dimensions) > 9:
            issues.append({
                "type": "dimension_limit",
                "message": f"Too many dimensions ({len(dimensions)}). GA4 allows maximum 9.",
                "suggestion": "Reduce the number of dimensions or run separate queries."
            })
            
        if len(metrics) > 10:
            issues.append({
                "type": "metric_limit", 
                "message": f"Too many metrics ({len(metrics)}). GA4 allows maximum 10.",
                "suggestion": "Reduce the number of metrics or run separate queries."
            })
        
        return {
            "valid": len(conflicts) == 0 and len(issues) == 0,
            "conflicts": conflicts,
            "issues": issues,
            "auto_resolved": len(conflicts) > 0  # Indicates if conflicts were auto-resolved
        }

    @classmethod
    def _get_default_query(cls, reason: str = "Unknown query") -> Dict:
        """Provide intelligent default with explanation"""
        return {
            "dimensions": ["date"],
            "metrics": ["sessions", "totalUsers", "screenPageViews"],
            "type": "traffic",
            "confidence": 0.2,
            "detected_intents": {},
            "query_interpretation": f"Using default traffic analysis - {reason}",
            "suggestion": "Try being more specific: 'Show traffic trends', 'Analyze page performance', 'Compare mobile vs desktop users', etc."
        }
    
    @staticmethod
    def parse_date_range(user_request: str) -> Tuple[str, str]:
        """Enhanced date range parsing"""
        request_lower = user_request.lower()
        
        date_patterns = [
            (["last 90 days", "past 3 months", "quarterly"], ("90daysAgo", "today")),
            (["last 60 days", "past 2 months"], ("60daysAgo", "today")),
            (["last 30 days", "past month", "monthly"], ("30daysAgo", "today")),
            (["last 14 days", "past 2 weeks"], ("14daysAgo", "today")),
            (["last 7 days", "past week", "weekly"], ("7daysAgo", "today")),
            (["yesterday"], ("yesterday", "yesterday")),
            (["today"], ("today", "today")),
            (["this week"], ("7daysAgo", "today")),
            (["this month"], ("30daysAgo", "today")),
        ]
        
        for patterns, date_range in date_patterns:
            if any(pattern in request_lower for pattern in patterns):
                return date_range
        
        return "30daysAgo", "today"  # Default to last 30 days

class ProfessionalVisualizer:
    """Enhanced visualization with professional styling"""
    
    def __init__(self):
        self.brand_colors = BRAND_COLORS
        # Modern, accessible color sequence for charts
        self.color_sequence = [
            self.brand_colors['primary'],    # Modern blue
            self.brand_colors['emerald'],    # Emerald green  
            self.brand_colors['purple'],     # Professional purple
            self.brand_colors['amber'],      # Amber yellow
            self.brand_colors['info'],       # Teal blue
            self.brand_colors['secondary'],  # Professional red
            self.brand_colors['indigo'],     # Rich indigo
            self.brand_colors['pink'],       # Modern pink
            self.brand_colors['success'],    # Clean green
            self.brand_colors['slate']       # Neutral slate
        ]
    
    def create_executive_summary(self, data: List[Dict], query_type: str) -> str:
        """Create executive summary with key insights"""
        if not data:
            if query_type == "engagement":
                return "**Executive Summary**: No engagement data available for analysis. This could indicate that your GA4 property doesn't have engagement tracking configured or there's no user activity in the selected time period."
            else:
                return "**Executive Summary**: No data available for analysis. Try adjusting your time period or check if your GA4 property has data for the requested metrics."
        
        df = pd.DataFrame(data)
        summary_parts = []
        
        # Time period analysis
        if 'date' in df.columns:
            date_range = f"{df['date'].min()} to {df['date'].max()}"
            summary_parts.append(f"**Analysis Period**: {date_range}")
        
        # Key metrics overview
        if 'totalUsers' in df.columns:
            total_users = df['totalUsers'].sum()
            summary_parts.append(f"**Total Users**: {total_users:,}")
        
        if 'sessions' in df.columns:
            total_sessions = df['sessions'].sum()
            summary_parts.append(f"**Total Sessions**: {total_sessions:,}")
            
            if 'totalUsers' in df.columns and total_users > 0:
                sessions_per_user = total_sessions / total_users
                summary_parts.append(f"**Sessions per User**: {sessions_per_user:.2f}")
        
        if 'screenPageViews' in df.columns:
            total_pageviews = df['screenPageViews'].sum()
            summary_parts.append(f"**Total Page Views**: {total_pageviews:,}")
        
        if 'conversions' in df.columns:
            total_conversions = df['conversions'].sum()
            summary_parts.append(f"**Total Conversions**: {total_conversions:,}")
            
            if 'totalUsers' in df.columns and total_users > 0:
                conversion_rate = (total_conversions / total_users) * 100
                summary_parts.append(f"**Conversion Rate**: {conversion_rate:.2f}%")
        
        # Engagement metrics summary
        if 'engagementRate' in df.columns:
            avg_engagement = df['engagementRate'].mean()
            summary_parts.append(f"**Avg Engagement Rate**: {avg_engagement:.2f}%")
            
        if 'bounceRate' in df.columns:
            avg_bounce = df['bounceRate'].mean()
            summary_parts.append(f"**Avg Bounce Rate**: {avg_bounce:.2f}%")
            
        if 'averageSessionDuration' in df.columns:
            avg_duration = df['averageSessionDuration'].mean()
            minutes = int(avg_duration // 60)
            seconds = int(avg_duration % 60)
            summary_parts.append(f"**Avg Session Duration**: {minutes}m {seconds}s")
        
        # Add query-type specific summaries
        if query_type == "page_performance" or query_type == "pages":
            if 'pagePath' in df.columns:
                unique_pages = df['pagePath'].nunique()
                summary_parts.append(f"**Pages Analyzed**: {unique_pages:,}")
        
        if not summary_parts:
            return f"**Executive Summary**: Data retrieved for {query_type} analysis but specific metrics are not available or contain zero values."
        
        return " | ".join(summary_parts)
    
    def create_traffic_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create modern, professional traffic trend chart"""
        fig = go.Figure()
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Add multiple metrics if available with modern colors
            metrics = [
                ('sessions', 'Sessions', self.brand_colors['primary']),
                ('totalUsers', 'Users', self.brand_colors['emerald']),
                ('screenPageViews', 'Page Views', self.brand_colors['purple'])
            ]
            
            for metric, label, color in metrics:
                if metric in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=df[metric],
                        mode='lines+markers',
                        name=label,
                        line=dict(color=color, width=4, shape='spline'),
                        marker=dict(size=8, color=color, line=dict(width=2, color='white')),
                        hovertemplate=f'<b>{label}</b><br>Date: %{{x|%b %d, %Y}}<br>Count: %{{y:,}}<extra></extra>',
                        # Removed problematic fill for now - keeping clean lines
                    ))
            
            fig.update_layout(
                title=dict(
                    text="📈 Website Traffic Trends",
                    font=dict(size=20, color='#1f2937', family='Arial, sans-serif', weight='bold'),
                    x=0.05
                ),
                xaxis_title="Date",
                yaxis_title="Visitors / Sessions",
                hovermode='x unified',
                template='simple_white',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12, color='#374151', family='Arial, sans-serif'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12, family='Arial, sans-serif'),
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#e5e7eb',
                    borderwidth=1
                ),
                margin=dict(t=100, l=60, r=40, b=60)
            )
            
            # Modern axes styling
            fig.update_xaxes(
                gridcolor='rgba(156, 163, 175, 0.2)',
                linecolor='#e5e7eb',
                tickfont=dict(size=11, color='#6b7280'),
                showgrid=True,
                zeroline=False
            )
            fig.update_yaxes(
                gridcolor='rgba(156, 163, 175, 0.2)',
                linecolor='#e5e7eb',
                tickfont=dict(size=11, color='#6b7280'),
                showgrid=True,
                zeroline=False
            )
        
        return fig
    
    def create_comparison_chart(self, df: pd.DataFrame, dimension: str, metric: str) -> go.Figure:
        """Create professional comparison charts"""
        # Take top 10 for readability
        df_sorted = df.nlargest(10, metric)
        
        # Create hover template with raw data if available
        if dimension == "pagePath" and f"{dimension}_raw" in df_sorted.columns:
            hover_template = '<b>%{y}</b><br>' + f'{metric.title()}: %{{x:,}}<br>' + 'Raw Path: %{customdata}<extra></extra>'
            customdata = df_sorted[f"{dimension}_raw"]
        else:
            hover_template = f'<b>%{{y}}</b><br>{metric.title()}: %{{x:,}}<extra></extra>'
            customdata = None
        
        fig = go.Figure(data=[
            go.Bar(
                x=df_sorted[metric],
                y=df_sorted[dimension],
                orientation='h',
                marker=dict(
                    color=df_sorted[metric],
                    colorscale='viridis',
                    colorbar=dict(title=metric.title())
                ),
                text=df_sorted[metric],
                texttemplate='%{text:,}',
                textposition='auto',
                hovertemplate=hover_template,
                customdata=customdata
            )
        ])
        
        fig.update_layout(
            title=dict(
                text=f"Top 10 {dimension.title()} by {metric.title()}",
                font=dict(size=16, color='#1e293b')
            ),
            xaxis_title=metric.title(),
            yaxis_title=dimension.title(),
            template='simple_white',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12, color='#374151'),
            margin=dict(l=150, t=60, r=40, b=60)
        )
        
        # Update axes styling
        fig.update_xaxes(
            gridcolor='#f1f5f9',
            linecolor='#e2e8f0',
            tickfont=dict(size=11, color='#6b7280')
        )
        fig.update_yaxes(
            gridcolor='#f1f5f9',
            linecolor='#e2e8f0',
            tickfont=dict(size=11, color='#6b7280')
        )
        
        return fig
    
    def create_pie_chart(self, df: pd.DataFrame, dimension: str, metric: str) -> go.Figure:
        """Create professional pie chart"""
        # Group small segments into "Others"
        df_sorted = df.sort_values(metric, ascending=False)
        
        if len(df_sorted) > 8:
            top_segments = df_sorted.head(7)
            others_value = df_sorted.tail(len(df_sorted) - 7)[metric].sum()
            
            # Add "Others" segment
            others_row = {dimension: 'Others', metric: others_value}
            if f"{dimension}_raw" in df_sorted.columns:
                others_row[f"{dimension}_raw"] = 'Others'
            top_segments = pd.concat([top_segments, pd.DataFrame([others_row])], ignore_index=True)
        else:
            top_segments = df_sorted
        
        # Create hover template with raw data if available
        if dimension == "pagePath" and f"{dimension}_raw" in top_segments.columns:
            hover_template = '<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<br>Raw Path: %{customdata}<extra></extra>'
            customdata = top_segments[f"{dimension}_raw"]
        else:
            hover_template = '<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
            customdata = None
        
        fig = go.Figure(data=[
            go.Pie(
                labels=top_segments[dimension],
                values=top_segments[metric],
                hole=0.4,  # Donut chart
                textinfo='label+percent',
                textposition='auto',
                marker=dict(colors=self.color_sequence * 3),  # Repeat colors if needed
                hovertemplate=hover_template,
                customdata=customdata
            )
        ])
        
        fig.update_layout(
            title=dict(
                text=f"{dimension.title()} Distribution by {metric.title()}",
                font=dict(size=16, color='#1e293b')
            ),
            template='simple_white',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12, color='#374151'),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05,
                font=dict(size=11)
            ),
            margin=dict(t=60, l=40, r=150, b=60)
        )
        
        return fig
    
    def create_modern_comparison_chart(self, df: pd.DataFrame, dimension: str, metric: str, chart_title: str) -> go.Figure:
        """Create modern, user-friendly comparison charts with separate bars"""
        
        # Debug: Log what we're working with
        logger.info(f"Creating chart for {dimension} vs {metric}")
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Original data columns: {df.columns.tolist()}")
        
        # CRITICAL: Log the actual raw data we're receiving
        logger.info("=== RAW DATA ANALYSIS ===")
        logger.info(f"First 5 rows of raw data:")
        logger.info(df.head().to_string())
        logger.info("=== END RAW DATA ===")
        
        # Check what's actually in the dataframe
        if dimension in df.columns:
            logger.info(f"Unique {dimension} values in raw data: {df[dimension].nunique()}")
            logger.info(f"Value counts for {dimension}:")
            logger.info(df[dimension].value_counts().head(10).to_string())
        
        # Check if there are additional dimensions that might cause duplication
        potential_grouping_columns = []
        for col in df.columns:
            if col not in [dimension, metric]:
                unique_vals = df[col].nunique()
                if unique_vals > 1:
                    potential_grouping_columns.append(f"{col}({unique_vals} values)")
        
        if potential_grouping_columns:
            logger.warning(f"Additional dimensions detected that might cause duplication: {potential_grouping_columns}")
            logger.warning("This might explain why you're seeing multiple bars per item")
            # Show a sample to understand the data structure
            logger.warning(f"Sample data rows: {df.head(3).to_dict('records')}")
        
        # Check for duplicate dimension values and log them
        if dimension in df.columns:
            value_counts = df[dimension].value_counts()
            duplicates = value_counts[value_counts > 1]
            if not duplicates.empty:
                logger.warning(f"Found duplicate {dimension} values:")
                for value, count in duplicates.head(5).items():
                    logger.warning(f"  '{value}' appears {count} times")
                    # Show sample rows for this value
                    sample_rows = df[df[dimension] == value][[dimension, metric]].head(3)
                    logger.warning(f"  Sample rows: {sample_rows.to_dict('records')}")
                    
                    # If there are additional dimensions, show them too
                    if len(df.columns) > 2:
                        full_sample = df[df[dimension] == value].head(3)
                        logger.warning(f"  Full sample rows: {full_sample.to_dict('records')}")
        
        # Clean and standardize dimension values before grouping
        df_clean = df.copy()
        if dimension == 'pagePath':
            # Standardize page paths
            df_clean[dimension] = df_clean[dimension].astype(str).str.strip()  # Remove whitespace
            df_clean[dimension] = df_clean[dimension].str.replace('//', '/')  # Fix double slashes
            df_clean[dimension] = df_clean[dimension].str.rstrip('/')  # Remove trailing slashes
            df_clean[dimension] = df_clean[dimension].replace('', '/')  # Empty strings become root
            logger.info(f"Cleaned pagePath values. Sample: {df_clean[dimension].head(3).tolist()}")
        elif dimension in ['source', 'deviceCategory', 'country']:
            # Clean other common dimensions
            df_clean[dimension] = df_clean[dimension].astype(str).str.strip()
            df_clean[dimension] = df_clean[dimension].str.replace('(not set)', 'Unknown')  # Standardize unknowns
        
        # IMPORTANT: Group by dimension and sum the metric to avoid duplicate bars
        logger.info("=== BEFORE GROUPING ===")
        logger.info(f"Data to be grouped (first 10 rows):")
        logger.info(df_clean[[dimension, metric]].head(10).to_string())
        
        df_grouped = df_clean.groupby(dimension)[metric].sum().reset_index()
        logger.info("=== AFTER GROUPING ===")
        logger.info(f"After grouping: {len(df_grouped)} unique {dimension} values")
        logger.info(f"Grouped data:")
        logger.info(df_grouped.to_string())
        
        # Log the grouping effect
        original_rows = len(df)
        grouped_rows = len(df_grouped)
        if original_rows != grouped_rows:
            logger.info(f"Data consolidation: {original_rows} rows -> {grouped_rows} rows (reduced by {original_rows - grouped_rows})")
        else:
            logger.warning("NO DATA CONSOLIDATION OCCURRED - This might explain multiple bars!")
        
        # Take top 10 for readability and sort
        df_sorted = df_grouped.nlargest(10, metric).reset_index(drop=True)
        
        # CRITICAL: Final validation that we have unique dimension values
        if df_sorted[dimension].duplicated().any():
            logger.error("FATAL: Still have duplicates after grouping! Emergency deduplication...")
            df_sorted = df_sorted.drop_duplicates(subset=[dimension], keep='first').reset_index(drop=True)
        
        logger.info(f"Final chart data: {df_sorted[[dimension, metric]].to_dict('records')}")
        logger.info(f"Dimension values to be charted: {df_sorted[dimension].tolist()}")
        
        # Create user-friendly labels
        dimension_labels = self._create_friendly_labels(df_sorted[dimension], dimension)
        logger.info(f"Friendly labels: {dimension_labels}")
        
        # Create clear, descriptive axis labels
        if metric == 'sessions':
            x_axis_label = "Number of Sessions"
            hover_label = "Sessions"
        elif metric == 'totalUsers':
            x_axis_label = "Number of Users"
            hover_label = "Users"
        elif metric == 'screenPageViews':
            x_axis_label = "Page Views"
            hover_label = "Page Views"
        else:
            x_axis_label = metric.replace('_', ' ').title()
            hover_label = metric
        
        # Create chart with proper sorting and NO duplicates
        fig = go.Figure()
        
        logger.info(f"Creating {len(df_sorted)} individual bar traces")
        logger.info("=== FINAL TRACE DATA ===")
        
        # Log exactly what we're about to chart
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            logger.info(f"Trace {i+1}: {row[dimension]} -> {dimension_labels[i]} = {row[metric]:,}")
        
        logger.info("=== CREATING PLOTLY TRACES ===")
        
        # Create individual traces for each UNIQUE dimension value
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            color = self.color_sequence[i % len(self.color_sequence)]
            
            logger.info(f"Creating Plotly trace {i+1}: y=['{dimension_labels[i]}'], x=[{row[metric]}]")
            
            fig.add_trace(go.Bar(
                x=[row[metric]],  # Single value as list
                y=[dimension_labels[i]],  # Single label as list
                orientation='h',
                name=f"{dimension_labels[i]}_{i}",  # Unique name to prevent conflicts
                marker=dict(
                    color=color,
                    line=dict(color='rgba(255,255,255,0.8)', width=1)
                ),
                text=[f'{row[metric]:,}'],
                textposition='auto',
                textfont=dict(size=11, color='white', family='Arial, sans-serif'),
                hovertemplate=f'<b>{dimension_labels[i]}</b><br>{hover_label}: {row[metric]:,}<extra></extra>',
                showlegend=False  # Hide legend for cleaner look
            ))
        
        logger.info(f"=== FINAL RESULT: Created {len(fig.data)} Plotly traces ===")
        
        # Create very clear subtitle explaining exactly what we're showing
        if dimension == 'deviceCategory':
            subtitle = f"📊 Total sessions per device type | Showing {len(df_sorted)} device categories"
        elif dimension == 'source':
            subtitle = f"📊 Total sessions per traffic source | Showing {len(df_sorted)} sources"
        elif dimension == 'country':
            subtitle = f"📊 Total sessions per country | Showing {len(df_sorted)} countries"
        elif dimension == 'pagePath':
            metric_name = "page views" if metric == 'screenPageViews' else "sessions"
            subtitle = f"📊 Total {metric_name} per page | Showing {len(df_sorted)} pages"
        else:
            subtitle = f"📊 Total {metric} per {dimension.replace('_', ' ').lower()} | Showing {len(df_sorted)} items"
        
        title_with_subtitle = f"{chart_title}<br><span style='font-size:12px; color:#6b7280;'>{subtitle}</span>"
        
        fig.update_layout(
            title=dict(
                text=title_with_subtitle,
                font=dict(size=18, color='#1f2937', family='Arial, sans-serif', weight='bold'),
                x=0.05
            ),
            xaxis_title=x_axis_label,
            yaxis_title="Device Types" if dimension == 'deviceCategory' else dimension.replace('_', ' ').title(),
            template='simple_white',
            height=max(400, len(df_sorted) * 50 + 120),  # More space per bar for clarity
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12, color='#374151', family='Arial, sans-serif'),
            margin=dict(l=200, t=100, r=60, b=60),
            showlegend=False,
            bargap=0.3,  # Add gap between bars for clarity
        )
        
        # Modern axes styling
        fig.update_xaxes(
            gridcolor='rgba(156, 163, 175, 0.2)',
            linecolor='#e5e7eb',
            tickfont=dict(size=11, color='#6b7280'),
            showgrid=True,
            zeroline=False
        )
        fig.update_yaxes(
            gridcolor='rgba(0,0,0,0)',
            linecolor='#e5e7eb',
            tickfont=dict(size=11, color='#374151', family='Arial, sans-serif'),
            showgrid=False,
            categoryorder='total ascending',  # Sort bars by value (smallest to largest)
            type='category'  # Ensure categorical axis
        )
        
        return fig
    
    def _create_friendly_labels(self, labels, dimension_type):
        """Create user-friendly labels for different dimension types"""
        friendly_labels = []
        
        for label in labels:
            if dimension_type == 'deviceCategory':
                # Capitalize device categories
                if label.lower() == 'mobile':
                    friendly_labels.append('📱 Mobile')
                elif label.lower() == 'desktop':
                    friendly_labels.append('💻 Desktop')
                elif label.lower() == 'tablet':
                    friendly_labels.append('📟 Tablet')
                else:
                    friendly_labels.append(f'📱 {label.title()}')
            
            elif dimension_type == 'source':
                # Clean up traffic source names
                if label.lower() == 'google':
                    friendly_labels.append('🔍 Google')
                elif label.lower() == 'direct':
                    friendly_labels.append('🔗 Direct')
                elif label.lower() == 'facebook.com':
                    friendly_labels.append('📘 Facebook')
                elif label.lower() == 'twitter.com':
                    friendly_labels.append('🐦 Twitter')
                elif label.lower() == 'linkedin.com':
                    friendly_labels.append('💼 LinkedIn')
                elif '(not set)' in str(label):
                    friendly_labels.append('❓ Unknown Source')
                else:
                    friendly_labels.append(f'🌐 {label}')
            
            elif dimension_type == 'country':
                # Add flag emojis for countries (basic ones)
                country_flags = {
                    'united states': '🇺🇸', 'usa': '🇺🇸', 'us': '🇺🇸',
                    'united kingdom': '🇬🇧', 'uk': '🇬🇧', 'britain': '🇬🇧',
                    'canada': '🇨🇦', 'germany': '🇩🇪', 'france': '🇫🇷',
                    'japan': '🇯🇵', 'australia': '🇦🇺', 'india': '🇮🇳',
                    'china': '🇨🇳', 'brazil': '🇧🇷', 'italy': '🇮🇹',
                    'spain': '🇪🇸', 'mexico': '🇲🇽', 'russia': '🇷🇺'
                }
                flag = country_flags.get(str(label).lower(), '🌍')
                friendly_labels.append(f'{flag} {label.title()}')
            
            elif dimension_type == 'pagePath':
                # Show the FULL page path for complete clarity
                path = str(label)
                if path in ['/', '', '/index.html', '/home']:
                    friendly_labels.append('🏠 Homepage')
                else:
                    # Show the complete path - no truncation, maximum clarity
                    friendly_labels.append(f'📄 {path}')
            
            else:
                # Default formatting
                clean_label = str(label).replace('(not set)', 'Unknown').title()
                if len(clean_label) > 30:
                    friendly_labels.append(clean_label[:27] + '...')
                else:
                    friendly_labels.append(clean_label)
        
        return friendly_labels
    
    def create_engagement_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create engagement metrics chart"""
        fig = go.Figure()
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Add engagement metrics if available
            engagement_metrics = [
                ('engagementRate', 'Engagement Rate (%)', self.brand_colors['primary']),
                ('bounceRate', 'Bounce Rate (%)', self.brand_colors['danger']),
                ('averageSessionDuration', 'Avg Session Duration (s)', self.brand_colors['success'])
            ]
            
            for metric, label, color in engagement_metrics:
                if metric in df.columns:
                    # For session duration, convert to minutes for better readability
                    if metric == 'averageSessionDuration':
                        y_values = df[metric] / 60  # Convert to minutes
                        y_label = 'Avg Session Duration (min)'
                        hover_template = f'<b>{label}</b><br>Date: %{{x}}<br>Duration: %{{y:.1f}} min<extra></extra>'
                    else:
                        y_values = df[metric]
                        y_label = label
                        hover_template = f'<b>{label}</b><br>Date: %{{x}}<br>Rate: %{{y:.2f}}%<extra></extra>'
                    
                    fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=y_values,
                        mode='lines+markers',
                        name=y_label,
                        line=dict(color=color, width=3),
                        marker=dict(size=6, color=color),
                        hovertemplate=hover_template
                    ))
            
            fig.update_layout(
                title=dict(
                    text="User Engagement Metrics",
                    font=dict(size=16, color='#1e293b')
                ),
                xaxis_title="Date",
                yaxis_title="Values",
                hovermode='x unified',
                template='simple_white',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12, color='#374151'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=11)
                ),
                margin=dict(t=60, l=60, r=40, b=60)
            )
            
            # Update axes styling
            fig.update_xaxes(
                gridcolor='#f1f5f9',
                linecolor='#e2e8f0',
                tickfont=dict(size=11, color='#6b7280')
            )
            fig.update_yaxes(
                gridcolor='#f1f5f9',
                linecolor='#e2e8f0',
                tickfont=dict(size=11, color='#6b7280')
        )
        
        return fig

class ReportGenerator:
    """Professional report generation"""
    
    def __init__(self, ga_manager: GA4DataManager):
        self.ga_manager = ga_manager
        self.visualizer = ProfessionalVisualizer()
    
    def generate_comprehensive_report(self, user_request: str) -> Dict:
        """Generate a comprehensive professional report"""
        try:
            # Parse the request with robust intelligence
            query_config = RobustQueryIntelligence.parse_user_request(user_request)
            date_range = RobustQueryIntelligence.parse_date_range(user_request)
            
            # Safety check: Ensure query_config is not None
            if query_config is None:
                logger.error("Query config is None - parse_user_request failed")
                return {
                    "success": False,
                    "error": "Failed to parse user request",
                    "suggestion": "Please try rephrasing your request or use one of the example queries"
                }
            
            # Safety check: Ensure required keys exist
            if 'dimensions' not in query_config or 'metrics' not in query_config:
                logger.error(f"Query config missing required keys: {query_config}")
                return {
                    "success": False,
                    "error": "Invalid query configuration",
                    "suggestion": "Please try a different query or use one of the example queries"
                }
            
            # Safety check: Ensure date_range is valid
            if date_range is None or len(date_range) != 2:
                logger.error(f"Invalid date_range: {date_range}")
                date_range = ("30daysAgo", "today")  # Fallback to default
            
            logger.info(f"Processing query: {user_request}")
            logger.info(f"Query config: {query_config}")
            logger.info(f"Date range: {date_range}")
            
            # Get data
            data_result = self.ga_manager.get_data_with_retry(
                dimensions=query_config['dimensions'],
                metrics=query_config['metrics'],
                start_date=date_range[0],
                end_date=date_range[1],
                limit=10000
            )
            
            if not data_result['success']:
                # Enhanced error handling with GA4 compatibility info
                error_response = {
                    "success": False,
                    "error": data_result['error'],
                    "suggestion": data_result.get('suggestion', 'Please try a different query')
                }
                
                # Add compatibility information if available
                if 'compatibility_info' in data_result:
                    error_response['compatibility_info'] = data_result['compatibility_info']
                
                # Check if this is a GA4 compatibility error and provide helpful suggestions
                if '400' in str(data_result.get('error', '')) and 'incompatible' in str(data_result.get('error', '')).lower():
                    # Log the failed query for debugging
                    logger.warning(f"GA4 compatibility error detected. Dimensions: {query_config.get('dimensions')}, Metrics: {query_config.get('metrics')}")
                    
                    error_response['suggestion'] = (
                        "This appears to be a GA4 API compatibility issue. "
                        "Try using fewer metrics or dimensions, or choose different combinations. "
                        "The system has built-in conflict resolution, but some edge cases may still occur."
                    )
                    
                    # Add specific debugging info
                    error_response['debug_info'] = {
                        "attempted_dimensions": query_config.get('dimensions'),
                        "attempted_metrics": query_config.get('metrics'),
                        "conflicts_resolved": query_config.get('conflicts_resolved'),
                        "original_metrics": query_config.get('original_metrics'),
                        "original_dimensions": query_config.get('original_dimensions')
                    }
                
                return error_response
            
            # Safety check: Ensure data_result has required keys
            if 'data' not in data_result:
                logger.error(f"Data result missing 'data' key: {data_result}")
                return {
                    "success": False,
                    "error": "Invalid data response from GA4 API",
                    "suggestion": "Please try again or check your GA4 property settings"
                }
            
            df = pd.DataFrame(data_result['data'])
            
            # Safety check: Ensure query_config has 'type' key
            query_type = query_config.get('type', 'traffic')  # Default to 'traffic' if missing
            
            # Generate insights
            insights = self._generate_insights(df, query_type)
            
            # Create visualizations
            visualizations = self._create_visualizations(df, query_config)
            
            return {
                "success": True,
                "executive_summary": self.visualizer.create_executive_summary(data_result['data'], query_type),
                "insights": insights,
                "visualizations": visualizations,
                "data": data_result['data'],
                "metadata": {
                    **data_result.get('metadata', {}),  # Safe access with default empty dict
                    "query_interpretation": query_config.get('query_interpretation'),
                    "confidence": query_config.get('confidence'),
                    "detected_intents": query_config.get('detected_intents'),
                    "suggestion": query_config.get('suggestion'),
                    "conflicts_resolved": query_config.get('conflicts_resolved'),
                    "original_metrics": query_config.get('original_metrics'),
                    "original_dimensions": query_config.get('original_dimensions'),
                    "metric_conflicts_resolved": query_config.get('metric_conflicts_resolved'),
                    "dimension_conflicts_resolved": query_config.get('dimension_conflicts_resolved'),
                    "emergency_fix_applied": query_config.get('emergency_fix_applied'),
                    "metrics": query_config.get('metrics'),
                    "dimensions": query_config.get('dimensions')
                }
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "suggestion": "Please try simplifying your request or check your data access"
            }
    
    def _generate_insights(self, df: pd.DataFrame, query_type: str) -> List[str]:
        """Generate data-driven insights"""
        insights = []
        
        # Check for empty data first
        if df.empty:
            return ["📊 No data available for the selected time period and metrics", 
                   "💡 Try selecting a different date range or check if your GA4 property has data for these metrics"]
        
        if query_type == "traffic" and 'date' in df.columns:
            # Trend analysis
            if len(df) > 7:
                recent_week = df.tail(7)['sessions'].mean() if 'sessions' in df.columns else 0
                previous_week = df.iloc[-14:-7]['sessions'].mean() if len(df) >= 14 and 'sessions' in df.columns else recent_week
                
                if previous_week > 0:
                    change = ((recent_week - previous_week) / previous_week) * 100
                    trend = "increased" if change > 0 else "decreased"
                    insights.append(f"📈 Traffic has {trend} by {abs(change):.1f}% week-over-week")
            
            # Peak performance
            if 'sessions' in df.columns:
                peak_day = df.loc[df['sessions'].idxmax()]
                insights.append(f"🔥 Peak traffic day: {peak_day.get('date', 'N/A')} with {peak_day['sessions']:,} sessions")
        
        elif query_type == "acquisition" and 'source' in df.columns:
            top_source = df.loc[df['sessions'].idxmax()] if 'sessions' in df.columns else None
            if top_source is not None:
                insights.append(f"🎯 Top traffic source: {top_source['source']} ({top_source['sessions']:,} sessions)")
        
        elif (query_type == "pages" or query_type == "page_performance") and 'pagePath' in df.columns:
            # Use the available metric (screenPageViews or sessions)
            metric_col = 'screenPageViews' if 'screenPageViews' in df.columns else 'sessions'
            metric_name = "views" if metric_col == 'screenPageViews' else "sessions"
            
            if metric_col in df.columns:
                top_page = df.loc[df[metric_col].idxmax()]
                page_name = top_page['pagePath']
                # Clean up page name for display
                display_name = page_name.replace('/', '').replace('-', ' ').title() if page_name else 'Unknown Page'
                if display_name == '' or display_name == 'Index':
                    display_name = 'Homepage'
                
                insights.append(f"📄 Top performing page: {display_name} ({top_page[metric_col]:,} {metric_name})")
                
                # Total pages analyzed
                unique_pages = df['pagePath'].nunique()
                insights.append(f"📊 Analyzed {unique_pages} different pages")
                
                # Performance distribution
                if len(df) > 2:
                    total_metric = df[metric_col].sum()
                    top_3_metric = df.nlargest(3, metric_col)[metric_col].sum()
                    top_3_percentage = (top_3_metric / total_metric) * 100
                    insights.append(f"🎯 Top 3 pages account for {top_3_percentage:.1f}% of total {metric_name}")
                
                # Homepage specific insights if available
                homepage_variants = ['/', '/index.html', '/home', 'Homepage', '']
                homepage_data = df[df['pagePath'].isin(homepage_variants)]
                if not homepage_data.empty:
                    homepage_metric = homepage_data[metric_col].sum()
                    homepage_percentage = (homepage_metric / total_metric) * 100
                    insights.append(f"🏠 Homepage accounts for {homepage_percentage:.1f}% of all page {metric_name}")
        
        elif query_type == "engagement":
            # Engagement metrics analysis
            if 'engagementRate' in df.columns:
                avg_engagement = df['engagementRate'].mean()
                insights.append(f"🎯 Average engagement rate: {avg_engagement:.2f}%")
                
                if 'date' in df.columns and len(df) > 1:
                    best_day = df.loc[df['engagementRate'].idxmax()]
                    insights.append(f"📈 Best engagement day: {best_day.get('date', 'N/A')} ({best_day['engagementRate']:.2f}%)")
            
            if 'bounceRate' in df.columns:
                avg_bounce = df['bounceRate'].mean()
                insights.append(f"⚡ Average bounce rate: {avg_bounce:.2f}%")
            
            if 'averageSessionDuration' in df.columns:
                avg_duration = df['averageSessionDuration'].mean()
                minutes = int(avg_duration // 60)
                seconds = int(avg_duration % 60)
                insights.append(f"⏱️ Average session duration: {minutes}m {seconds}s")
        
        elif query_type == "geographic":
            # Geographic analysis insights
            if 'country' in df.columns and 'sessions' in df.columns:
                top_country = df.loc[df['sessions'].idxmax()]
                total_sessions = df['sessions'].sum()
                top_percentage = (top_country['sessions'] / total_sessions) * 100
                insights.append(f"🌍 Top country: {top_country['country']} ({top_percentage:.1f}% of traffic)")
                
                unique_countries = df['country'].nunique()
                insights.append(f"🗺️ Traffic from {unique_countries} different countries")
        
        elif query_type == "ecommerce":
            # Ecommerce analysis insights
            if 'itemsPurchased' in df.columns:
                total_items = df['itemsPurchased'].sum()
                insights.append(f"🛒 Total items purchased: {total_items:,}")
                
                if 'itemName' in df.columns:
                    top_product = df.loc[df['itemsPurchased'].idxmax()]
                    insights.append(f"🏆 Best selling product: {top_product['itemName']} ({top_product['itemsPurchased']:,} units)")
            
            if 'itemRevenue' in df.columns:
                total_revenue = df['itemRevenue'].sum()
                insights.append(f"💰 Total product revenue: ${total_revenue:,.2f}")
        
        elif query_type == "conversion":
            # Conversion analysis insights
            if 'conversions' in df.columns:
                total_conversions = df['conversions'].sum()
                insights.append(f"🎯 Total conversions: {total_conversions:,}")
                
                if 'totalRevenue' in df.columns and total_conversions > 0:
                    avg_order_value = df['totalRevenue'].sum() / total_conversions
                    insights.append(f"💵 Average order value: ${avg_order_value:.2f}")
        
        elif query_type == "source":
            # Traffic source insights (alias for acquisition)
            if 'source' in df.columns and 'sessions' in df.columns:
                top_source = df.loc[df['sessions'].idxmax()]
                insights.append(f"🎯 Top traffic source: {top_source['source']} ({top_source['sessions']:,} sessions)")
                
                if 'sessionDefaultChannelGroup' in df.columns:
                    top_channel = df.groupby('sessionDefaultChannelGroup')['sessions'].sum().idxmax()
                    insights.append(f"📢 Top channel: {top_channel}")
        
        elif query_type == "device":
            # Device analysis insights (alias for technology)
            if 'deviceCategory' in df.columns and 'sessions' in df.columns:
                device_breakdown = df.groupby('deviceCategory')['sessions'].sum()
                total_sessions = device_breakdown.sum()
                
                for device, sessions in device_breakdown.head(3).items():
                    percentage = (sessions / total_sessions) * 100
                    insights.append(f"📱 {device.title()}: {percentage:.1f}% of sessions")
        
        # If no insights were generated for any query type, provide a generic message
        if not insights:
            insights.append("📊 Data retrieved successfully, but no specific insights could be generated")
            insights.append("💡 Try using different metrics or time periods for more detailed analysis")
        
        return insights
    
    def _create_visualizations(self, df: pd.DataFrame, query_config: Dict) -> List[Dict]:
        """Create comprehensive, user-friendly visualizations"""
        visualizations = []
        
        # Check for empty data first
        if df.empty:
            return []
        
        # Check if this is a comprehensive query
        query_interpretation = query_config.get('query_interpretation', '').lower()
        detected_intents = query_config.get('detected_intents', {})
        is_comprehensive = (
            len(detected_intents) > 1 or 
            'comprehensive' in query_interpretation or
            'dashboard' in query_interpretation or
            ('traffic' in query_interpretation and 'sources' in query_interpretation) or
            ('traffic' in query_interpretation and 'device' in query_interpretation) or
            ('including' in query_interpretation and len(query_interpretation.split('and')) > 2)
        )
        
        logger.info(f"Query routing - Comprehensive: {is_comprehensive}, Query type: {query_config.get('type')}")
        logger.info(f"Detected intents: {detected_intents}")
        logger.info(f"Query interpretation: {query_interpretation}")
        
        if is_comprehensive:
            # Create multiple charts for comprehensive analysis
            logger.info("ROUTING: Using comprehensive dashboard")
            visualizations = self._create_comprehensive_dashboard(df)
        else:
            # Create single focused visualization
            logger.info(f"ROUTING: Using single visualization for type: {query_config.get('type')}")
            if query_config['type'] == "traffic" and 'date' in df.columns:
                fig = self.visualizer.create_traffic_chart(df)
                visualizations.append({"type": "traffic_trend", "figure": fig})
            
            elif query_config['type'] == "acquisition":
                if 'source' in df.columns and 'sessions' in df.columns:
                    fig = self.visualizer.create_modern_comparison_chart(df, 'source', 'sessions', 'Traffic Sources')
                    visualizations.append({"type": "source_comparison", "figure": fig})
            
            elif query_config['type'] == "technology" or query_config['type'] == "device":
                if 'deviceCategory' in df.columns and 'sessions' in df.columns:
                    # Use modern bar chart instead of pie chart for better readability
                    fig = self.visualizer.create_modern_comparison_chart(df, 'deviceCategory', 'sessions', 'Device Usage')
                    visualizations.append({"type": "device_distribution", "figure": fig})
            
            elif query_config['type'] == "pages" or query_config['type'] == "page_performance":
                # Determine which metric to use (prefer sessions if available, otherwise use screenPageViews)
                if 'pagePath' in df.columns:
                    metric_to_use = 'sessions' if 'sessions' in df.columns else 'screenPageViews'
                    if metric_to_use in df.columns:
                        fig = self.visualizer.create_modern_comparison_chart(df, 'pagePath', metric_to_use, '📄 Top Performing Pages')
                        visualizations.append({"type": "page_performance", "figure": fig})
            
            elif query_config['type'] == "engagement":
                if 'date' in df.columns:
                    fig = self.visualizer.create_engagement_chart(df)
                    visualizations.append({"type": "engagement_trends", "figure": fig})
            
            elif query_config['type'] == "geographic":
                if 'country' in df.columns and 'sessions' in df.columns:
                    fig = self.visualizer.create_modern_comparison_chart(df, 'country', 'sessions', 'Geographic Distribution')
                    visualizations.append({"type": "geographic_distribution", "figure": fig})
        
        # Fallback: Create best available visualization
        if not visualizations and not df.empty:
            if 'date' in df.columns:
                fig = self.visualizer.create_traffic_chart(df)
                visualizations.append({"type": "general_trends", "figure": fig})
            else:
                # Find first suitable dimension and metric
                dimensions = [col for col in df.columns if col not in ['sessions', 'totalUsers']]
                if dimensions and 'sessions' in df.columns:
                    fig = self.visualizer.create_modern_comparison_chart(df, dimensions[0], 'sessions', 'Website Analytics')
                    visualizations.append({"type": "general_comparison", "figure": fig})
        
        return visualizations
    
    def _create_comprehensive_dashboard(self, df: pd.DataFrame) -> List[Dict]:
        """Create multiple charts for comprehensive website analytics"""
        visualizations = []
        
        logger.info(f"Creating comprehensive dashboard with columns: {df.columns.tolist()}")
        
        # 1. Traffic Trends (if date data available)
        if 'date' in df.columns and any(col in df.columns for col in ['sessions', 'totalUsers']):
            fig = self.visualizer.create_traffic_chart(df)
            visualizations.append({"type": "traffic_trends", "figure": fig})
        
        # 2. Device Usage (if device data available)
        if 'deviceCategory' in df.columns and 'sessions' in df.columns:
            fig = self.visualizer.create_modern_comparison_chart(df, 'deviceCategory', 'sessions', '📱 Device Usage Analysis')
            visualizations.append({"type": "device_usage", "figure": fig})
        
        # 3. Traffic Sources (if source data available)
        if 'source' in df.columns and 'sessions' in df.columns:
            fig = self.visualizer.create_modern_comparison_chart(df, 'source', 'sessions', '🌐 Traffic Sources Analysis')
            visualizations.append({"type": "traffic_sources", "figure": fig})
        
        # 4. Popular Pages (if page data available)
        if 'pagePath' in df.columns:
            # Use sessions if available, otherwise use screenPageViews
            page_metric = 'sessions' if 'sessions' in df.columns else 'screenPageViews'
            if page_metric in df.columns:
                chart_title = '📄 Popular Pages Analysis' if page_metric == 'sessions' else '📄 Top Performing Pages Analysis'
                fig = self.visualizer.create_modern_comparison_chart(df, 'pagePath', page_metric, chart_title)
                visualizations.append({"type": "popular_pages", "figure": fig})
        
        # 5. Geographic Distribution (if country data available)
        if 'country' in df.columns and 'sessions' in df.columns:
            fig = self.visualizer.create_modern_comparison_chart(df, 'country', 'sessions', '🌍 Geographic Distribution')
            visualizations.append({"type": "geographic", "figure": fig})
            
        # 6. User Behavior (if engagement data available)
        if any(col in df.columns for col in ['engagementRate', 'bounceRate', 'averageSessionDuration']) and 'date' in df.columns:
            fig = self.visualizer.create_engagement_chart(df)
            visualizations.append({"type": "user_engagement", "figure": fig})
        
        # If no specific charts could be created, create a general overview
        if not visualizations:
            logger.warning("No specific comprehensive charts could be created, falling back to available data")
            if 'sessions' in df.columns:
                # Find the best dimension to chart against
                available_dims = [col for col in df.columns if col not in ['sessions', 'totalUsers', 'date']]
                if available_dims:
                    best_dim = available_dims[0]
                    fig = self.visualizer.create_modern_comparison_chart(df, best_dim, 'sessions', '📊 Website Analytics Overview')
                    visualizations.append({"type": "general_overview", "figure": fig})
        
        logger.info(f"Created {len(visualizations)} visualizations for comprehensive dashboard")
        return visualizations

# Enhanced AutoGen Integration
def create_enhanced_analyst_agent():
    """Create enhanced analyst agent with better prompting"""
    return autogen.AssistantAgent(
        name="ga_analyst",
        llm_config=llm_config,
        system_message="""You are a senior Google Analytics consultant providing professional analysis for business clients.

CORE RESPONSIBILITIES:
1. Analyze GA4 data with business context and strategic insights
2. Identify actionable opportunities and potential issues  
3. Provide clear, executive-level recommendations
4. Structure analysis professionally for client presentation

ANALYSIS FRAMEWORK:
- Executive Summary (key metrics and timeframe)
- Key Findings (data-driven insights)
- Performance Analysis (trends, patterns, anomalies)
- Strategic Recommendations (specific, actionable steps)
- Next Steps (suggested follow-up analysis)

PROFESSIONAL STANDARDS:
- Use business language, not technical jargon
- Focus on ROI and business impact
- Provide context for all metrics
- Include confidence levels for predictions
- Suggest specific optimization opportunities

CRITICAL: Always structure your response with clear sections and actionable insights."""
    )

# Streamlit App Configuration
def configure_streamlit():
    """Configure Streamlit for professional appearance"""
    st.set_page_config(
        page_title="GA4 Analytics Intelligence Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }
    
    .main-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a202c;
        text-align: center;
        margin-bottom: 0.4rem;
        letter-spacing: -0.01em;
    }
    
    .subtitle {
        font-size: 0.9rem;
        font-weight: 400;
        color: #718096;
        text-align: center;
        margin-bottom: 1.25rem;
    }
    
    .metric-card {
        background: #ffffff;
        padding: 0.8rem;
        border-radius: 4px;
        border: 1px solid #d1d5db;
        margin: 0.4rem 0;
        font-size: 0.8rem;
        line-height: 1.3;
        color: #374151;
        font-weight: 500;
    }
    
    .insight-box {
        background: #f9fafb;
        padding: 0.7rem;
        border-radius: 4px;
        border-left: 1px solid #d1d5db;
        margin: 0.3rem 0;
        font-size: 0.8rem;
        line-height: 1.3;
        color: #6b7280;
    }
    
    .stButton > button {
        background-color: #4299e1;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.45rem 0.9rem;
        font-weight: 500;
        font-size: 0.85rem;
        max-width: 280px;
        margin: 0 auto;
        display: block;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    .stButton > button:hover {
        background-color: #3182ce;
        border: none;
    }
    
    .sidebar .stButton > button {
        max-width: none;
        width: 100%;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        font-size: 0.85rem;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    .stTextArea > div > div > textarea {
        font-size: 0.85rem;
        line-height: 1.4;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    /* Headers - Professional and compact */
    h1, h2, h3 {
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        letter-spacing: 0;
    }
    
    h2 {
        font-size: 0.65rem;
        color: #4a5568;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    h3 {
        font-size: 0.9rem;
        color: #4a5568;
        margin-bottom: 0.4rem;
        font-weight: 500;
    }
    
    /* Selectbox and other inputs */
    .stSelectbox > div > div {
        font-size: 0.85rem;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 0.75rem;
        border-radius: 6px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    
    /* Tips section - More subtle */
    .tips-section {
        background: #f7fafc;
        padding: 0.85rem;
        border-radius: 6px;
        border-left: 2px solid #cbd5e0;
        font-size: 0.8rem;
        line-height: 1.3;
        color: #4a5568;
    }
    
    /* Chart containers with borders */
    .js-plotly-plot {
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        overflow: hidden !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #718096;
        font-size: 0.75rem;
        margin-top: 2.5rem;
        padding: 0.75rem;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Remove default margins */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    /* General text improvements */
    p, div, span {
        font-family: 'Source Sans Pro', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    configure_streamlit()
    
    # Header
    st.markdown('<h1 class="main-header">GA4 Analytics Intelligence Platform</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Transform your Google Analytics data into actionable business insights</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # GA4 Property ID
        property_id = st.text_input(
            "GA4 Property ID",
            value=os.getenv("GA4_PROPERTY_ID", ""),
            help="Your Google Analytics 4 Property ID (numbers only)"
        )
        
        # Service Account Status
        service_account_path = "service-account.json"
        if os.path.exists(service_account_path):
            st.success("✅ Service Account: Connected")
        else:
            st.error("❌ Service Account: Not Found")
            st.info("Add your service-account.json file to the project directory")
        
        # Connection Test
        if st.button("🔍 Test Connection", use_container_width=True):
            if property_id and os.path.exists(service_account_path):
                ga_manager = GA4DataManager(property_id)
                with st.spinner("Testing connection..."):
                    result = ga_manager.authenticate()
                    if result["success"]:
                        st.success("✅ " + result["message"])
                    else:
                        st.error("❌ " + result["error"])
                        if "suggestion" in result:
                            st.info("💡 " + result["suggestion"])
            else:
                st.warning("⚠️ Please configure Property ID and service account")
        
        st.divider()
        
        # Quick Query Examples
        st.subheader("💡 Example Queries")
        example_queries = [
            # Basic queries
            "Show me traffic trends for the last 30 days",
            "Analyze top performing pages this month", 
            "Compare mobile vs desktop users",
            "What are my main traffic sources?",
            "Show conversion performance last week",
            "Analyze user engagement metrics",
            
            # Advanced queries
            "Which countries bring the most valuable users?",
            "Show me product performance by category",
            "Compare bounce rates across different devices",
            "What's my revenue trend over the past quarter?", 
            "Which campaigns are driving the most conversions?",
            "How do new vs returning users behave differently?",
            "Show me geographic distribution of my audience",
            "Analyze shopping cart abandonment patterns",
            "Which pages have the longest session duration?",
            "Compare weekend vs weekday traffic patterns"
        ]
        
        selected_example = st.selectbox(
            "Quick Examples:",
            [""] + example_queries,
            help="Select an example query to get started"
        )
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # User input with default prompt
        default_prompt = "Show me comprehensive website analytics for the last 30 days including traffic trends, top pages, traffic sources, and device usage"
        
        # Check if an alternative query was selected from error page
        query_value = default_prompt
        if selected_example:
            query_value = selected_example
        elif hasattr(st.session_state, 'alternative_query') and st.session_state.alternative_query:
            query_value = st.session_state.alternative_query
            # Clear the alternative query after using it
            st.session_state.alternative_query = None
        
        user_query = st.text_area(
            "What would you like to analyze?",
            value=query_value,
            height=120,
            placeholder="Example: 'Show me traffic trends for the last 30 days and identify my top traffic sources'"
        )
    
    with col2:
        st.markdown("### 🎯 Query Tips")
        st.markdown("""
        <div class="tips-section">
        <strong>Time Periods:</strong><br>
        • "last 30 days", "this month"<br>
        • "past week", "yesterday"<br><br>
        
        <strong>Analysis Types:</strong><br>
        • Traffic trends<br>
        • Page performance<br>
        • Traffic sources<br>
        • Device analysis<br>
        • Conversions<br><br>
        
        <strong>💡 Smart Features:</strong><br>
        • Auto-resolves GA4 compatibility issues<br>
        • Handles synonyms & variations<br>
        • Suggests alternatives for unclear queries
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize session state for report data
    if 'report_data' not in st.session_state:
        st.session_state.report_data = None
    if 'show_raw_data' not in st.session_state:
        st.session_state.show_raw_data = False
    
    # Generate Report Button - centered and appropriately sized
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        generate_report = st.button("📊 Create Report", type="primary")
    
    if generate_report:
        if not property_id:
            st.error("❌ Please enter your GA4 Property ID in the sidebar")
        elif not os.path.exists(service_account_path):
            st.error("❌ Service account file not found. Please add service-account.json to your project directory")
        elif not user_query.strip():
            st.error("❌ Please enter your analysis request")
        else:
            # Initialize components
            ga_manager = GA4DataManager(property_id)
            report_generator = ReportGenerator(ga_manager)
            
            # Generate report
            with st.spinner("🔄 Analyzing your data... This may take a moment"):
                report = report_generator.generate_comprehensive_report(user_query)
                st.session_state.report_data = report
                st.session_state.show_raw_data = False  # Reset raw data view
    
    # Display report if we have data (either from current generation or session state)
    if st.session_state.report_data and st.session_state.report_data["success"]:
        report = st.session_state.report_data
        
        # Debug information is now hidden from users but still logged for troubleshooting
        # Display Executive Summary
        st.markdown("## Executive Summary")
        st.markdown('<div class="metric-card">' + report["executive_summary"] + '</div>', unsafe_allow_html=True)
        
        # Display Key Insights
        st.markdown("## Key Insights")
        if report.get("insights"):
            for insight in report["insights"]:
                st.markdown('<div class="insight-box">' + insight + '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-box">📊 No specific insights available - this may indicate limited data for the selected metrics and time period.</div>', unsafe_allow_html=True)
        
        # Display Visualizations
            st.markdown("## Charts & Analysis")
        if report.get("visualizations"):
            for viz in report["visualizations"]:
                st.plotly_chart(viz["figure"], use_container_width=True)
        else:
            st.markdown('<div class="insight-box">📈 No charts available - insufficient data for visualization. Try adjusting your query or time period.</div>', unsafe_allow_html=True)
        
        # Data Export Section
            st.markdown("## Data Export")
        if report.get("data") and len(report["data"]) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV Export
                df = pd.DataFrame(report["data"])
                csv = df.to_csv(index=False)
                st.download_button(
                    "📄 Download CSV",
                    csv,
                    f"ga4_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # JSON Export
                json_data = json.dumps(report["data"], indent=2)
                st.download_button(
                    "📝 Download JSON",
                    json_data,
                    f"ga4_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    use_container_width=True
                )
            
            with col3:
                # Toggle for raw data view
                if st.button("👁️ View Raw Data", use_container_width=True):
                    st.session_state.show_raw_data = not st.session_state.show_raw_data
            
            # Show raw data if toggled
            if st.session_state.show_raw_data:
                st.markdown("### Raw Data")
                st.dataframe(df, use_container_width=True)
        else:
            st.markdown('<div class="insight-box">💾 No data available for export. The query returned empty results.</div>', unsafe_allow_html=True)
        

    
    elif st.session_state.report_data and not st.session_state.report_data["success"]:
        st.error(f"❌ Report generation failed: {st.session_state.report_data['error']}")
        
        if st.session_state.report_data.get("suggestion"):
            st.info(f"💡 Suggestion: {st.session_state.report_data['suggestion']}")
        
        # Show GA4 compatibility information if available
        if st.session_state.report_data.get("compatibility_info"):
            compat_info = st.session_state.report_data["compatibility_info"]
            
            if compat_info.get("conflicts"):
                st.warning("⚠️ **GA4 Compatibility Issues Detected:**")
                for conflict in compat_info["conflicts"]:
                    if conflict["type"] == "metric_conflict":
                        conflicting = ", ".join(conflict["conflicting_metrics"])
                        st.write(f"• Cannot use these metrics together: **{conflicting}**")
                        st.write(f"  *Suggestion: {conflict['suggestion']}*")
                    elif conflict["type"] == "dimension_metric_conflict":
                        metric = conflict["metric"]
                        dimensions = ", ".join(conflict["conflicting_dimensions"])
                        st.write(f"• Cannot use metric **{metric}** with dimensions: **{dimensions}**")
                        st.write(f"  *Reason: {conflict['reason']}*")
            
            if compat_info.get("auto_resolved"):
                st.success("✅ **Auto-Resolution Applied:** The system automatically resolved metric conflicts for you!")
        
        # Provide quick alternative suggestions
        st.markdown("### 🔧 Try These Alternatives:")
        alternative_queries = [
            "Show me basic traffic trends",
            "Analyze page performance only", 
            "Compare device categories",
            "Show user engagement metrics",
            "Analyze traffic sources"
        ]
        
        cols = st.columns(len(alternative_queries))
        for i, alt_query in enumerate(alternative_queries):
            with cols[i]:
                if st.button(f"📊 {alt_query}", key=f"alt_{i}", use_container_width=True):
                    st.session_state.alternative_query = alt_query
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div class="footer"><strong>GA4 Analytics Intelligence Platform</strong> | Powered by Google Analytics 4 API & AI Analysis</div>',
        unsafe_allow_html=True
    )

def test_ga4_compatibility():
    """Quick test of GA4 compatibility system"""
    print("Testing GA4 compatibility system...")
    
    # Test dimension-metric conflict resolution
    dimensions = ["userType", "pagePath"]  # Should conflict with screenPageViews
    metrics = ["screenPageViews", "sessions"]
    
    resolved_dims, resolved_metrics = RobustQueryIntelligence._resolve_dimension_metric_conflicts(dimensions, metrics)
    
    print(f"Original dimensions: {dimensions}")
    print(f"Original metrics: {metrics}")
    print(f"Resolved dimensions: {resolved_dims}")
    print(f"Resolved metrics: {resolved_metrics}")
    
    # Should have removed userType but kept pagePath
    assert "userType" not in resolved_dims, "Should have removed userType dimension"
    assert "pagePath" in resolved_dims, "Should have kept pagePath dimension"
    assert "screenPageViews" in resolved_metrics, "Should have kept screenPageViews metric"
    
    print("✅ Basic compatibility tests passed!")

if __name__ == "__main__":
    # Run basic compatibility tests to verify system is working (disabled in production)
    # Uncomment the lines below to run tests:
    # try:
    #     test_ga4_compatibility()
    # except Exception as e:
    #     print(f"Compatibility test failed: {e}")
    
    main()