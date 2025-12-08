import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
# streamlit run ujj.py
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import logging
from datetime import datetime
import seaborn as sns
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Portfolio Optimization App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define language options
languages = {
    'English': 'en',
    'हिन्दी': 'hi',
    'বাংলা': 'bn'
}

# Define language strings
translations = {
    'en': {
        "title": "Hybrid ML Models for Solving Interval Linear Systems and Portfolio Selection Under Uncertainty",
        "user_inputs": "🔧 User Inputs",
        "select_universe": "Select an Asset Universe:",
        "custom_tickers": "Enter stock tickers separated by commas (e.g., AAPL, MSFT, TSLA):",
        "add_portfolio": "Add to My Portfolio",
        "my_portfolio": "📁 My Portfolio",
        "no_assets": "No assets added yet.",
        "optimization_parameters": "📅 Optimization Parameters",
        "start_date": "Start Date",
        "end_date": "End Date",
        "risk_free_rate": "Enter the risk-free rate (in %):",
        "investment_strategy": "Choose your Investment Strategy:",
        "strategy_risk_free": "Risk-free Investment",
        "strategy_profit": "Profit-focused Investment",
        "target_return": "Select a specific target return (in %)",
        "train_lstm": "Train LSTM Model for Future Returns Prediction",
        "more_info_lstm": "ℹ️ More Information on LSTM",
        "optimize_portfolio": "Optimize Portfolio",
        "optimize_sharpe": "Optimize for Highest Sharpe Ratio",
        "compare_portfolios": "Compare Sharpe vs Base",
        "robust_optimize": "Robust Portfolio Optimization",
        "ilp_optimize": "Interval Linear Programming Optimization",
        "portfolio_analysis": "🔍 Portfolio Analysis & Optimization Results",
        "success_lstm": "🤖 LSTM model trained successfully!",
        "error_no_assets_lstm": "Please add at least one asset to your portfolio before training the LSTM model.",
        "error_no_assets_opt": "Please add at least one asset to your portfolio before optimization.",
        "error_date": "Start date must be earlier than end date.",
        "allocation_title": "🔑 Optimal Portfolio Allocation (Target Return: {target}%)",
        "performance_metrics": "📊 Portfolio Performance Metrics",
        "visual_analysis": "📊 Visual Analysis",
        "portfolio_composition": "Portfolio Composition",
        "portfolio_metrics": "Portfolio Performance Metrics",
        "correlation_heatmap": "Asset Correlation Heatmap",
        "var": "Value at Risk (VaR)",
        "cvar": "Conditional Value at Risk (CVaR)",
        "max_drawdown": "Maximum Drawdown",
        "hhi": "Herfindahl-Hirschman Index (HHI)",
        "sharpe_ratio": "Sharpe Ratio",
        "sortino_ratio": "Sortino Ratio",
        "calmar_ratio": "Calmar Ratio",
        "beta": "Beta",
        "alpha": "Alpha",
        "explanation_var": "**Value at Risk (VaR):** Estimates the maximum potential loss of a portfolio over a specified time frame at a given confidence level.",
        "explanation_cvar": "**Conditional Value at Risk (CVaR):** Measures the expected loss exceeding the VaR, providing insights into tail risk.",
        "explanation_max_drawdown": "**Maximum Drawdown:** Measures the largest peak-to-trough decline in the portfolio value, indicating the worst-case scenario.",
        "explanation_hhi": "**Herfindahl-Hirschman Index (HHI):** A diversification metric that measures the concentration of investments in a portfolio.",
        "explanation_sharpe_ratio": "**Sharpe Ratio:** Measures risk-adjusted returns, indicating how much excess return you receive for the extra volatility endured.",
        "explanation_sortino_ratio": "**Sortino Ratio:** Similar to the Sharpe Ratio but only considers downside volatility, providing a more targeted risk-adjusted return measure.",
        "explanation_calmar_ratio": "**Calmar Ratio:** Compares the portfolio's annualized return to its maximum drawdown, indicating return per unit of risk.",
        "explanation_beta": "**Beta:** Measures the portfolio's volatility relative to a benchmark index (e.g., S&P 500). A beta greater than 1 indicates higher volatility than the benchmark.",
        "explanation_alpha": "**Alpha:** Represents the portfolio's excess return relative to the expected return based on its beta. Positive alpha indicates outperformance.",
        "explanation_lstm": "**Explanation of LSTM Model:**\nLong Short-Term Memory (LSTM) is a type of artificial neural network used in machine learning. It is particularly effective for predicting sequences and time series data, such as stock returns. LSTM models can remember information over long periods, making them suitable for capturing trends and patterns in historical financial data. However, while LSTM can provide valuable insights, it's important to note that predictions are not guarantees and should be used in conjunction with other analysis methods.",
        "feedback_sharpe_good": "Great! A Sharpe Ratio above 1 indicates that your portfolio is generating good returns for the level of risk taken.",
        "feedback_sharpe_average": "Average. A Sharpe Ratio between 0.5 and 1 suggests that your portfolio returns are acceptable for the risk taken.",
        "feedback_sharpe_poor": "Poor. A Sharpe Ratio below 0.5 indicates that your portfolio may not be generating adequate returns for the level of risk taken. Consider diversifying your assets or adjusting your investment strategy.",
        "success_optimize": "Portfolio optimization completed successfully!",
        "explanation_sharpe_button": "**Optimize for Highest Sharpe Ratio:**\nThe Sharpe Ratio measures the performance of your portfolio compared to a risk-free asset, after adjusting for its risk. Optimizing for the highest Sharpe Ratio aims to achieve the best possible return for the level of risk you are willing to take. This helps in constructing a portfolio that maximizes returns while minimizing unnecessary risk.",
        "recommendation": "Based on the above metrics, the **{better_portfolio}** portfolio is recommended for better **{better_metric}**.",
        "robust_optimization": "🛡️ Robust Portfolio Optimization",
        "ilp_optimization": "📐 Interval Linear Programming (ILP)",
        "scenarios": "Market Scenarios",
        "bull_scenario": "Bull Market (+20% returns)",
        "bear_scenario": "Bear Market (-20% returns)",
        "volatile_scenario": "Volatile Market (High Volatility)",
        "robust_results": "Robust Optimization Results",
        "scenario_results": "Scenario Analysis Results",
        "best_case": "Best Case Return",
        "worst_case": "Worst Case Return",
        "ilp_results": "ILP Optimization Results",
        "weight_intervals": "Weight Intervals (min-max for each asset)",
        "return_interval": "Return Interval (min-max %)",
        "risk_interval": "Risk Interval (min-max %)",
        "explanation_robust": "**Robust Optimization:** Tests portfolio performance under different market scenarios (bull, bear, volatile) to ensure resilience.",
        "explanation_ilp": "**Interval Linear Programming:** Finds optimal weight ranges for each asset considering uncertainty in returns and risks.",
        "uncertainty_level": "Uncertainty Level (%)",
        "robust_constraint": "Robustness Constraint (max loss %)"
    },
    'hi': {
        "title": "अंतराल रैखिक प्रणालियों को हल करने और अनिश्चितता के तहत पोर्टफोलियो चयन के लिए हाइब्रिड एमएल मॉडल",
        "user_inputs": "🔧 उपयोगकर्ता इनपुट",
        "select_universe": "एक संपत्ति समूह चुनें:",
        "custom_tickers": "कॉमा से अलग स्टॉक टिकर दर्ज करें (उदा. AAPL, MSFT, TSLA):",
        "add_portfolio": "मेरे पोर्टफोलियो में जोड़ें",
        "my_portfolio": "📁 मेरा पोर्टफोलियो",
        "no_assets": "अभी तक कोई संपत्ति नहीं जोड़ी गई है।",
        "optimization_parameters": "📅 अनुकूलन मापदंड",
        "start_date": "आरंभ तिथि",
        "end_date": "समाप्ति तिथि",
        "risk_free_rate": "जोखिम-मुक्त दर दर्ज करें (% में):",
        "investment_strategy": "अपनी निवेश रणनीति चुनें:",
        "strategy_risk_free": "जोखिम-मुक्त निवेश",
        "strategy_profit": "लाभ-केंद्रित निवेश",
        "target_return": "एक विशिष्ट लक्ष्य रिटर्न चुनें (% में)",
        "train_lstm": "भविष्य के रिटर्न की भविष्यवाणी के लिए LSTM मॉडल प्रशिक्षित करें",
        "more_info_lstm": "ℹ️ LSTM पर अधिक जानकारी",
        "optimize_portfolio": "पोर्टफोलियो अनुकूलित करें",
        "optimize_sharpe": "उच्चतम शार्प अनुपात के लिए अनुकूलित करें",
        "compare_portfolios": "शार्प बनाम आधार की तुलना करें",
        "robust_optimize": "रोबस्ट पोर्टफोलियो ऑप्टिमाइजेशन",
        "ilp_optimize": "इंटरवल लीनियर प्रोग्रामिंग ऑप्टिमाइजेशन",
        "portfolio_analysis": "🔍 पोर्टफोलियो विश्लेषण और अनुकूलन परिणाम",
        "success_lstm": "🤖 LSTM मॉडल सफलतापूर्वक प्रशिक्षित हो गया!",
        "error_no_assets_lstm": "कृपया LSTM मॉडल को प्रशिक्षित करने से पहले अपने पोर्टफोलियो में कम से कम एक संपत्ति जोड़ें।",
        "error_no_assets_opt": "कृपया अनुकूलन से पहले अपने पोर्टफोलियो में कम से कम एक संपत्ति जोड़ें।",
        "error_date": "आरंभ तिथि समाप्ति तिथि से पहले होनी चाहिए।",
        "allocation_title": "🔑 इष्टतम पोर्टफोलियो आवंटन (लक्ष्य रिटर्न: {target}%)",
        "performance_metrics": "📊 पोर्टफोलियो प्रदर्शन मेट्रिक्स",
        "visual_analysis": "📊 दृश्य विश्लेषण",
        "portfolio_composition": "पोर्टफोलियो संरचना",
        "portfolio_metrics": "पोर्टफोलियो प्रदर्शन मेट्रिक्स",
        "correlation_heatmap": "संपत्ति सहसंबंध हीटमैप",
        "var": "जोखिम मूल्य (VaR)",
        "cvar": "सशर्त जोखिम मूल्य (CVaR)",
        "max_drawdown": "अधिकतम गिरावट",
        "hhi": "हर्फिंडल-हिर्शमैन सूचकांक (HHI)",
        "sharpe_ratio": "शार्प अनुपात",
        "sortino_ratio": "सोर्टिनो अनुपात",
        "calmar_ratio": "कैल्मर अनुपात",
        "beta": "बीटा",
        "alpha": "अल्फा",
        "success_optimize": "पोर्टफोलियो अनुकूलन सफलतापूर्वक पूर्ण हुआ!",
        "recommendation": "उपरोक्त मेट्रिक्स के आधार पर, बेहतर **{better_metric}** के लिए **{better_portfolio}** पोर्टफोलियो की सिफारिश की जाती है।",
        "robust_optimization": "🛡️ रोबस्ट पोर्टफोलियो ऑप्टिमाइजेशन",
        "ilp_optimization": "📐 इंटरवल लीनियर प्रोग्रामिंग (ILP)",
        "scenarios": "मार्केट सीनारियो",
        "bull_scenario": "बुल मार्केट (+20% रिटर्न)",
        "bear_scenario": "बेयर मार्केट (-20% रिटर्न)",
        "volatile_scenario": "वोलेटाइल मार्केट (उच्च अस्थिरता)",
        "robust_results": "रोबस्ट ऑप्टिमाइजेशन रिजल्ट्स",
        "scenario_results": "सीनारियो विश्लेषण परिणाम",
        "best_case": "सर्वश्रेष्ठ स्थिति रिटर्न",
        "worst_case": "सबसे खराब स्थिति रिटर्न",
        "ilp_results": "ILP ऑप्टिमाइजेशन रिजल्ट्स",
        "weight_intervals": "वेट इंटरवल्स (प्रत्येक एसेट के लिए मिन-मैक्स)",
        "return_interval": "रिटर्न इंटरवल (मिन-मैक्स %)",
        "risk_interval": "रिस्क इंटरवल (मिन-मैक्स %)",
        "explanation_robust": "**रोबस्ट ऑप्टिमाइजेशन:** विभिन्न बाजार परिदृश्यों में पोर्टफोलियो प्रदर्शन का परीक्षण करता है।",
        "explanation_ilp": "**इंटरवल लीनियर प्रोग्रामिंग:** अनिश्चितता को ध्यान में रखते हुए इष्टतम वजन सीमा खोजता है।",
        "uncertainty_level": "अनिश्चितता स्तर (%)",
        "robust_constraint": "रोबस्टनेस बाधा (अधिकतम नुकसान %)",
        "explanation_lstm": "**LSTM मॉडल की व्याख्या:**\nLSTM (Long Short-Term Memory) एक प्रकार का न्यूरल नेटवर्क है जो समय-श्रृंखला डेटा जैसे स्टॉक रिटर्न की भविष्यवाणी के लिए उपयोग होता है। यह लंबे समय तक जानकारी याद रख सकता है और ट्रेंड तथा पैटर्न पकड़ने में मदद करता है। लेकिन यह केवल अनुमान देता है, गारंटी नहीं।"
    },
    'bn': {
        "title": "ব্যবধান রৈখিক সিস্টেম সমাধান এবং অনিশ্চয়তার অধীনে পোর্টফোলিও নির্বাচনের জন্য হাইব্রিড এমএল মডেল",
        "user_inputs": "🔧 ব্যবহারকারী ইনপুট",
        "select_universe": "একটি সম্পদ ইউনিভার্স নির্বাচন করুন:",
        "custom_tickers": "কমা দ্বারা পৃথক করা স্টক টিকার লিখুন (যেমন AAPL, MSFT, TSLA):",
        "add_portfolio": "আমার পোর্টফোলিওতে যোগ করুন",
        "my_portfolio": "📁 আমার পোর্টফোলিও",
        "no_assets": "এখনো কোনো সম্পদ যোগ করা হয়নি।",
        "optimization_parameters": "📅 অপ্টিমাইজেশন প্যারামিটার",
        "start_date": "শুরুর তারিখ",
        "end_date": "শেষ তারিখ",
        "risk_free_rate": "ঝুঁকি-মুক্ত হার লিখুন (% এ):",
        "investment_strategy": "আপনার বিনিয়োগ কৌশল নির্বাচন করুন:",
        "strategy_risk_free": "ঝুঁকি-মুক্ত বিনিয়োগ",
        "strategy_profit": "লাভ-কেন্দ্রিক বিনিয়োগ",
        "target_return": "একটি নির্দিষ্ট লক্ষ্য রিটার্ন নির্বাচন করুন (% এ)",
        "train_lstm": "ভবিষ্যত রিটার্ন পূর্বাভাসের জন্য LSTM মডেল প্রশিক্ষণ দিন",
        "more_info_lstm": "ℹ️ LSTM সম্পর্কে আরও তথ্য",
        "optimize_portfolio": "পোর্টফোলিও অপ্টিমাইজ করুন",
        "optimize_sharpe": "সর্বোচ্চ শার্প অনুপাতের জন্য অপ্টিমাইজ করুন",
        "compare_portfolios": "শার্প বনাম বেস তুলনা করুন",
        "robust_optimize": "রোবাস্ট পোর্টফোলিও অপ্টিমাইজেশন",
        "ilp_optimize": "ইন্টারভাল লিনিয়ার প্রোগ্রামিং অপ্টিমাইজেশন",
        "portfolio_analysis": "🔍 পোর্টফোলিও বিশ্লেষণ এবং অপ্টিমাইজেশন ফলাফল",
        "success_lstm": "🤖 LSTM মডেল সফলভাবে প্রশিক্ষিত হয়েছে!",
        "error_no_assets_lstm": "LSTM মডেল প্রশিক্ষণ দেওয়ার আগে আপনার পোর্টফোলিওতে কমপক্ষে একটি সম্পদ যোগ করুন।",
        "error_no_assets_opt": "অপ্টিমাইজেশনের আগে আপনার পোর্টফোলিওতে কমপক্ষে একটি সম্পদ যোগ করুন।",
        "error_date": "শুরুর তারিখ অবশ্যই শেষ তারিখের আগে হতে হবে।",
        "allocation_title": "🔑 সর্বোত্তম পোর্টফোলিও বরাদ্দ (লক্ষ্য রিটার্ন: {target}%)",
        "performance_metrics": "📊 পোর্টফোলিও পারফরম্যান্স মেট্রিক্স",
        "visual_analysis": "📊 ভিজ্যুয়াল বিশ্লেষণ",
        "portfolio_composition": "পোর্টফোলিও গঠন",
        "portfolio_metrics": "পোর্টফোলিও পারফরম্যান্স মেট্রিক্স",
        "correlation_heatmap": "সম্পদ সম্পর্ক হিটম্যাপ",
        "var": "ঝুঁকি মূল্য (VaR)",
        "cvar": "শর্তযুক্ত ঝুঁকি মূল্য (CVaR)",
        "max_drawdown": "সর্বোচ্চ পতন",
        "hhi": "হার্ফিন্ডাল-হির্শম্যান সূচক (HHI)",
        "sharpe_ratio": "শার্প অনুপাত",
        "sortino_ratio": "সর্টিনো অনুপাত",
        "calmar_ratio": "ক্যালমার অনুপাত",
        "beta": "বিটা",
        "alpha": "আলফা",
        "success_optimize": "পোর্টফোলিও অপ্টিমাইজেশন সফলভাবে সম্পন্ন হয়েছে!",
        "recommendation": "উপরের মেট্রিক্সের উপর ভিত্তি করে, আরও ভাল **{better_metric}** এর জন্য **{better_portfolio}** পোর্টফোলিও সুপারিশ করা হয়।",
        "robust_optimization": "🛡️ রোবাস্ট পোর্টফোলিও অপ্টিমাইজেশন",
        "ilp_optimization": "📐 ইন্টারভাল লিনিয়ার প্রোগ্রামিং (ILP)",
        "scenarios": "মার্কেট সিনারিও",
        "bull_scenario": "বুল মার্কেট (+20% রিটার্ন)",
        "bear_scenario": "বিয়ার মার্কেট (-20% রিটার্ন)",
        "volatile_scenario": "ভোলাটাইল মার্কেট (উচ্চ অস্থিরতা)",
        "robust_results": "রোবাস্ট অপ্টিমাইজেশন রেজাল্টস",
        "scenario_results": "সিনারিও বিশ্লেষণ ফলাফল",
        "best_case": "সেরা ক্ষেত্রে রিটার্ন",
        "worst_case": "সবচেয়ে খারাপ ক্ষেত্রে রিটার্ন",
        "ilp_results": "ILP অপ্টিমাইজেশন রেজাল্টস",
        "weight_intervals": "ওয়েট ইন্টারভালস (প্রতিটি অ্যাসেটের জন্য মিন-ম্যাক্স)",
        "return_interval": "রিটার্ন ইন্টারভাল (মিন-ম্যাক্স %)",
        "risk_interval": "রিস্ক ইন্টারভাল (মিন-ম্যাক্স %)",
        "explanation_robust": "**রোবাস্ট অপ্টিমাইজেশন:** বিভিন্ন বাজার পরিস্থিতিতে পোর্টফোলিও কর্মক্ষমতা পরীক্ষা করে।",
        "explanation_ilp": "**ইন্টারভাল লিনিয়ার প্রোগ্রামিং:** অনিশ্চয়তা বিবেচনা করে সর্বোত্তম ওজন পরিসীমা খুঁজে পায়।",
        "uncertainty_level": "অনিশ্চয়তার স্তর (%)",
        "robust_constraint": "রোবাস্টনেস সীমাবদ্ধতা (সর্বোচ্চ ক্ষতি %)",
        "explanation_lstm": "**LSTM মডেলের ব্যাখ্যা:**\nLSTM হলো একটি নিউরাল নেটওয়ার্ক যা টাইম সিরিজ ডেটা (যেমন স্টক রিটার্ন) পূর্বাভাসে ভালো কাজ করে। এটি দীর্ঘ সময়ের তথ্য মনে রাখতে পারে, তবে এটি অনুমান, গ্যারান্টি নয়।"
    }
}

#########################################################
# INTERVAL ARITHMETIC + N×N INTERVAL LINEAR SYSTEM SOLVER
#########################################################

class Interval:
    """
    μ-ρ representation: [μ-ρ, μ+ρ]
    """
    def __init__(self, mu: float, rho: float):
        self.mu = float(mu)
        self.rho = abs(float(rho))

    @property
    def lower(self):
        return self.mu - self.rho

    @property
    def upper(self):
        return self.mu + self.rho

    @staticmethod
    def from_bounds(lower: float, upper: float):
        lower = float(lower)
        upper = float(upper)
        if lower > upper:
            lower, upper = upper, lower
        mu = 0.5 * (lower + upper)
        rho = 0.5 * (upper - lower)
        return Interval(mu, rho)

    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self.mu + other.mu, self.rho + other.rho)
        return Interval(self.mu + float(other), self.rho)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(self.mu - other.mu, self.rho + other.rho)
        return Interval(self.mu - float(other), self.rho)

    def __rsub__(self, other):
        return Interval(float(other) - self.mu, self.rho)

    def _bounds_mul(self, other):
        a, b = self.lower, self.upper
        c, d = other.lower, other.upper
        cand = [a * c, a * d, b * c, b * d]
        return min(cand), max(cand)

    def __mul__(self, other):
        if isinstance(other, Interval):
            lo, hi = self._bounds_mul(other)
            return Interval.from_bounds(lo, hi)
        else:
            other = float(other)
            lo = self.lower * other
            hi = self.upper * other
            return Interval.from_bounds(min(lo, hi), max(lo, hi))

    __rmul__ = __mul__

    def reciprocal(self):
        if self.lower <= 0.0 <= self.upper:
            raise ValueError("Interval contains 0 → reciprocal undefined.")
        lo = 1.0 / self.upper
        hi = 1.0 / self.lower
        return Interval.from_bounds(min(lo, hi), max(lo, hi))

    def __truediv__(self, other):
        if isinstance(other, Interval):
            return self * other.reciprocal()
        else:
            other = float(other)
            if other == 0:
                raise ValueError("Division by zero.")
            return self * (1.0 / other)

    def __repr__(self):
        return f"[{self.lower:.4f}, {self.upper:.4f}]"


class IntervalVector:
    def __init__(self, intervals):
        self.intervals = [
            i if isinstance(i, Interval) else Interval(float(i), 0.0)
            for i in intervals
        ]
        self.n = len(self.intervals)

    def __getitem__(self, idx):
        return self.intervals[idx]

    @property
    def midpoints(self):
        return np.array([iv.mu for iv in self.intervals], dtype=float)

    @property
    def radii(self):
        return np.array([iv.rho for iv in self.intervals], dtype=float)


class IntervalMatrix:
    def __init__(self, matrix):
        self.matrix = []
        for row in matrix:
            row_int = []
            for elem in row:
                if isinstance(elem, Interval):
                    row_int.append(elem)
                else:
                    row_int.append(Interval(float(elem), 0.0))
            self.matrix.append(row_int)
        self.rows = len(self.matrix)
        self.cols = len(self.matrix[0]) if self.rows > 0 else 0

    def __getitem__(self, idx):
        return self.matrix[idx]

    @property
    def midpoint_matrix(self):
        return np.array([[c.mu for c in row] for row in self.matrix], dtype=float)

    @property
    def radius_matrix(self):
        return np.array([[c.rho for c in row] for row in self.matrix], dtype=float)


class IntervalLinearSystemSolver:
    """
    Hansen-style n×n interval linear system solver:
    Ã x̃ = b̃ with midpoint A_mid, radius A_rad, etc.
    """

    @staticmethod
    def hansen(A_mid, A_rad, b_mid, b_rad, max_iter=50, tol=1e-6):
        n = len(b_mid)

        # Initial guess: solve midpoint system
        try:
            x_mid = np.linalg.solve(A_mid, b_mid)
        except np.linalg.LinAlgError:
            x_mid = np.zeros(n)
        x_rho = np.ones(n) * 10.0

        # Preconditioner C ≈ A_mid^{-1}
        try:
            C = np.linalg.inv(A_mid)
        except np.linalg.LinAlgError:
            C = np.eye(n)

        for _ in range(max_iter):
            # midpoint residual
            r_mid = b_mid - A_mid @ x_mid
            x_mid_new = x_mid + C @ r_mid

            # simplified radius update
            term1 = np.abs(C) @ (A_rad @ np.abs(x_mid) + b_rad)
            x_rho_new = np.abs(np.eye(n) - C @ A_mid) @ x_rho + term1

            if (
                np.max(np.abs(x_mid_new - x_mid)) < tol
                and np.max(np.abs(x_rho_new - x_rho)) < tol
            ):
                x_mid, x_rho = x_mid_new, x_rho_new
                break

            x_mid, x_rho = x_mid_new, x_rho_new

        return IntervalVector([Interval(x_mid[i], abs(x_rho[i])) for i in range(n)])

    @staticmethod
    def solve(A: IntervalMatrix, b: IntervalVector):
        A_mid = A.midpoint_matrix
        A_rad = A.radius_matrix
        b_mid = b.midpoints
        b_rad = b.radii
        return IntervalLinearSystemSolver.hansen(A_mid, A_rad, b_mid, b_rad)

##############################
# Portfolio Optimizer Class
##############################
class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02):
        """
        Initialize the PortfolioOptimizer with user-specified parameters.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None

    def fetch_data(self):
        """
        Fetch historical price data and calculate daily returns.
        """
        logger.info(f"Fetching data for tickers: {self.tickers}")
        data = yf.download(
            self.tickers, start=self.start_date, end=self.end_date, progress=False
        )
        logger.info(f"Data columns: {data.columns}")
        st.subheader("Raw Fetched Data Preview:")
        st.dataframe(data.head())

        # Handle MultiIndex vs single-level
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.levels[0]:
                data = data.xs('Adj Close', axis=1, level=0)
            elif 'Close' in data.columns.levels[0]:
                data = data.xs('Close', axis=1, level=0)
            else:
                st.error("Neither 'Adj Close' nor 'Close' columns are available in multi-index.")
                raise ValueError("Neither 'Adj Close' nor 'Close' columns are available in multi-index.")
        else:
            if 'Adj Close' in data.columns:
                data = data['Adj Close']
            elif 'Close' in data.columns:
                data = data['Close']
            else:
                st.error("Neither 'Adj Close' nor 'Close' columns are available.")
                raise ValueError("Neither 'Adj Close' nor 'Close' columns are available.")

        data.dropna(axis=1, how='all', inplace=True)

        if data.empty:
            logger.error("No data fetched after dropping missing tickers.")
            raise ValueError("No data fetched. Please check the tickers and date range.")

        if isinstance(data, pd.DataFrame):
            self.tickers = list(data.columns)
        else:
            self.tickers = [data.name]
            data = pd.DataFrame(data)

        self.returns = data.pct_change().dropna()
        self.mean_returns = self.returns.mean() * 252
        self.cov_matrix = self.returns.cov() * 252
        logger.info(f"Fetched returns for {len(self.tickers)} tickers.")

        st.subheader("Processed Returns Dataset:")
        st.dataframe(self.returns.head(10))

        return self.tickers

    def portfolio_stats(self, weights):
        """
        Calculate portfolio return, volatility, and Sharpe ratio.
        """
        weights = np.array(weights)
        if len(weights) != len(self.tickers):
            raise ValueError("Weights array length does not match the number of tickers.")

        weights = weights / np.sum(weights)

        portfolio_return = np.dot(weights, self.mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def value_at_risk(self, weights, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR) for the portfolio.
        """
        portfolio_returns = self.returns.dot(weights)
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        return var

    def conditional_value_at_risk(self, weights, confidence_level=0.95):
        """
        Calculate Conditional Value at Risk (CVaR) for the portfolio.
        """
        portfolio_returns = self.returns.dot(weights)
        var = self.value_at_risk(weights, confidence_level)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return cvar

    def maximum_drawdown(self, weights):
        """
        Calculate Maximum Drawdown for the portfolio.
        """
        portfolio_returns = self.returns.dot(weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        return max_drawdown

    def herfindahl_hirschman_index(self, weights):
        """
        Calculate Herfindahl-Hirschman Index (HHI) for the portfolio.
        """
        return np.sum(weights ** 2)

    def sharpe_ratio_objective(self, weights):
        """
        Objective function to maximize Sharpe Ratio.
        """
        _, _, sharpe = self.portfolio_stats(weights)
        return -sharpe

    def optimize_sharpe_ratio(self):
        """
        Optimize portfolio to maximize Sharpe Ratio.
        """
        num_assets = len(self.tickers)
        initial_weights = np.ones(num_assets) / num_assets
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        result = minimize(
            self.sharpe_ratio_objective, initial_weights,
            method='SLSQP', bounds=bounds, constraints=constraints
        )

        if result.success:
            logger.info("Optimized portfolio for Sharpe Ratio successfully.")
            return result.x
        else:
            logger.warning(f"Optimization failed: {result.message}")
            return initial_weights

    def min_volatility(self, target_return, max_weight=0.3):
        """
        Optimize portfolio with added weight constraints for minimum volatility.
        """
        num_assets = len(self.tickers)
        constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
            {'type': 'eq', 'fun': lambda weights: self.portfolio_stats(weights)[0] - target_return}
        )
        bounds = tuple((0, max_weight) for _ in range(num_assets))
        init_guess = [1. / num_assets] * num_assets

        result = minimize(
            lambda weights: self.portfolio_stats(weights)[1],
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            logger.info("Optimized portfolio for minimum volatility successfully.")
            return result.x
        else:
            logger.warning(f"Portfolio optimization failed: {result.message}")
            return np.ones(num_assets) / num_assets

    def robust_optimization(self, uncertainty_level=0.1):
        """
        Robust Portfolio Optimization using worst-case scenario approach.
        Tests portfolio under bull, bear, and volatile market scenarios.
        """
        num_assets = len(self.tickers)
        scenarios = self.generate_market_scenarios(uncertainty_level)

        def robust_objective(weights):
            worst_sharpe = float('inf')
            for scenario in scenarios:
                mean_ret = scenario['mean_returns']
                cov_mat = scenario['cov_matrix']
                port_return = np.dot(weights, mean_ret)
                port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
                sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
                worst_sharpe = min(worst_sharpe, sharpe)
            return -worst_sharpe

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x}
        ]

        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = np.ones(num_assets) / num_assets

        result = minimize(
            robust_objective, initial_weights,
            method='SLSQP', bounds=bounds, constraints=constraints
        )

        if result.success:
            logger.info("Robust optimization completed successfully.")
            return result.x, scenarios
        else:
            logger.warning(f"Robust optimization failed: {result.message}")
            return initial_weights, scenarios

    def generate_market_scenarios(self, uncertainty_level=0.1):
        """
        Generate different market scenarios for robust optimization.
        """
        scenarios = []

        # Bull Market
        bull_mean = self.mean_returns * (1 + uncertainty_level)
        bull_cov = self.cov_matrix * (1 - uncertainty_level * 0.5)
        scenarios.append({'name': 'Bull Market', 'mean_returns': bull_mean, 'cov_matrix': bull_cov})

        # Bear Market
        bear_mean = self.mean_returns * (1 - uncertainty_level)
        bear_cov = self.cov_matrix * (1 + uncertainty_level)
        scenarios.append({'name': 'Bear Market', 'mean_returns': bear_mean, 'cov_matrix': bear_cov})

        # Volatile Market
        volatile_mean = self.mean_returns
        volatile_cov = self.cov_matrix * (1 + uncertainty_level * 1.5)
        scenarios.append({'name': 'Volatile Market', 'mean_returns': volatile_mean, 'cov_matrix': volatile_cov})

        return scenarios

    def interval_linear_programming(self, uncertainty_level=0.1):
        """
        Interval-based portfolio optimization.
        Mean returns and covariance are modeled as intervals, and we derive
        weight intervals and (return, risk) intervals.
        """
        num_assets = len(self.tickers)

        mean_returns_lower = self.mean_returns * (1 - uncertainty_level)
        mean_returns_upper = self.mean_returns * (1 + uncertainty_level)

        cov_matrix_lower = self.cov_matrix * (1 - uncertainty_level)
        cov_matrix_upper = self.cov_matrix * (1 + uncertainty_level)

        def best_case_objective(weights):
            port_return = np.dot(weights, mean_returns_upper)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_lower, weights)))
            return -(port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        def worst_case_objective(weights):
            port_return = np.dot(weights, mean_returns_lower)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_upper, weights)))
            return -(port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = np.ones(num_assets) / num_assets

        best_result = minimize(
            best_case_objective, initial_weights,
            method='SLSQP', bounds=bounds, constraints=constraints
        )

        worst_result = minimize(
            worst_case_objective, initial_weights,
            method='SLSQP', bounds=bounds, constraints=constraints
        )

        if best_result.success and worst_result.success:
            logger.info("ILP (interval-based) optimization completed successfully.")

            weight_intervals = []
            for i in range(num_assets):
                min_weight = min(best_result.x[i], worst_result.x[i])
                max_weight = max(best_result.x[i], worst_result.x[i])
                weight_intervals.append((min_weight, max_weight))

            best_return = np.dot(best_result.x, mean_returns_upper)
            worst_return = np.dot(worst_result.x, mean_returns_lower)

            best_risk = np.sqrt(np.dot(best_result.x.T, np.dot(cov_matrix_lower, best_result.x)))
            worst_risk = np.sqrt(np.dot(worst_result.x.T, np.dot(cov_matrix_upper, worst_result.x)))

            return {
                'best_weights': best_result.x,
                'worst_weights': worst_result.x,
                'weight_intervals': weight_intervals,
                'return_interval': (worst_return, best_return),
                'risk_interval': (best_risk, worst_risk)
            }
        else:
            logger.warning("ILP optimization failed.")
            return None

    def prepare_data_for_lstm(self):
        """
        Prepare data for LSTM model.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.returns.values)

        X, y = [], []
        look_back = 60
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i - look_back:i])
            y.append(scaled_data[i])

        if len(X) < 10:
            raise ValueError("Not enough data to create training samples. Please adjust the date range or add more data.")

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)
        return X_train, y_train, X_test, y_test, scaler

    def train_lstm_model(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Train LSTM model.
        """
        seed_value = 42
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        random.seed(seed_value)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True,
                                       input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(tf.keras.layers.LSTM(units=50))
        model.add(tf.keras.layers.Dense(units=X_train.shape[2]))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        return model

    def predict_future_returns(self, model, scaler, steps=30):
        """
        Predict multi-step future returns using the LSTM model.

        We take the last 60 days, then iteratively predict the next 'steps' days.
        """
        if len(self.returns) < 60:
            raise ValueError("Not enough data to make predictions. Ensure there are at least 60 days of returns data.")

        last_data = self.returns[-60:].values
        scaled_last_data = scaler.transform(last_data)

        history = scaled_last_data.copy()
        future_preds_scaled = []

        for _ in range(steps):
            X_input = history[-60:].reshape(1, 60, history.shape[1])
            next_scaled = model.predict(X_input, verbose=0)[0]  # shape (n_assets,)
            future_preds_scaled.append(next_scaled)
            history = np.vstack([history, next_scaled])

        future_preds_scaled = np.array(future_preds_scaled)  # (steps, n_assets)
        future_preds = scaler.inverse_transform(future_preds_scaled)
        return future_preds  # time x assets

    def evaluate_model(self, model, scaler, X_test, y_test):
        """
        Evaluate the LSTM model using MAE, RMSE, and R-squared metrics.
        """
        predictions_scaled = model.predict(X_test, verbose=0)
        predictions = scaler.inverse_transform(predictions_scaled)
        y_test_inverse = scaler.inverse_transform(y_test)

        mae = mean_absolute_error(y_test_inverse, predictions)
        rmse = np.sqrt(mean_squared_error(y_test_inverse, predictions))
        r2 = r2_score(y_test_inverse, predictions)

        return mae, rmse, r2

    def compute_efficient_frontier(self, num_portfolios=10000):
        """
        Compute the Efficient Frontier by generating random portfolios.
        """
        results = np.zeros((4, num_portfolios))
        weights_record = []
        for i in range(num_portfolios):
            weights = np.random.dirichlet(np.ones(len(self.tickers)), size=1)[0]
            weights_record.append(weights)
            portfolio_return, portfolio_volatility, sharpe = self.portfolio_stats(weights)
            results[0, i] = portfolio_volatility
            results[1, i] = portfolio_return
            results[2, i] = sharpe
            results[3, i] = self.herfindahl_hirschman_index(weights)
        return results, weights_record


##############################
# Helper Functions
##############################
def extract_ticker(asset_string):
    """
    Extract ticker symbol from asset string.
    """
    return asset_string.split(' - ')[0].strip() if ' - ' in asset_string else asset_string.strip()

def get_translated_text(lang, key):
    """
    Retrieve translated text based on selected language.
    """
    return translations.get(lang, translations['en']).get(key, key)

def analyze_var(var):
    """
    Analyze Value at Risk (VaR).
    """
    if var < -0.05:
        return "High Risk: Your portfolio has a significant potential loss."
    elif -0.05 <= var < -0.02:
        return "Moderate Risk: Your portfolio has a moderate potential loss."
    else:
        return "Low Risk: Your portfolio is relatively safe."

def analyze_cvar(cvar):
    """
    Analyze Conditional Value at Risk (CVaR).
    """
    if cvar < -0.07:
        return "High Tail Risk: Significant losses beyond VaR."
    elif -0.07 <= cvar < -0.04:
        return "Moderate Tail Risk: Moderate losses beyond VaR."
    else:
        return "Low Tail Risk: Minimal losses beyond VaR."

def analyze_max_drawdown(dd):
    """
    Analyze Maximum Drawdown.
    """
    if dd < -0.20:
        return "Severe Drawdown: The portfolio has experienced a major decline."
    elif -0.20 <= dd < -0.10:
        return "Moderate Drawdown: The portfolio has experienced a noticeable decline."
    else:
        return "Minor Drawdown: The portfolio has maintained stability."

def analyze_hhi(hhi):
    """
    Analyze Herfindahl-Hirschman Index (HHI).
    """
    if hhi > 0.6:
        return "High Concentration: Portfolio lacks diversification."
    elif 0.3 < hhi <= 0.6:
        return "Moderate Concentration: Portfolio has some diversification."
    else:
        return "Good Diversification: Portfolio is well-diversified."

def analyze_sharpe(sharpe):
    """
    Analyze Sharpe Ratio.
    """
    if sharpe > 1:
        return "Great! A Sharpe Ratio above 1 indicates that your portfolio is generating good returns for the level of risk taken."
    elif 0.5 < sharpe <= 1:
        return "Average. A Sharpe Ratio between 0.5 and 1 suggests that your portfolio returns are acceptable for the risk taken."
    else:
        return "Poor. A Sharpe Ratio below 0.5 indicates that your portfolio may not be generating adequate returns for the level of risk taken."

def display_metrics_table(metrics, lang):
    """
    Display metrics in a structured table.
    """
    metric_display = []
    for key, value in metrics.items():
        display_key = get_translated_text(lang, key)
        if key in ["hhi"]:
            display_value = f"{value:.4f}"
        elif key in ["beta", "alpha"]:
            display_value = f"{value:.2f}"
        elif key in ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]:
            display_value = f"{value:.2f}"
        else:
            display_value = f"{value:.2%}"

        analysis_func = {
            "var": analyze_var,
            "cvar": analyze_cvar,
            "max_drawdown": analyze_max_drawdown,
            "hhi": analyze_hhi,
            "sharpe_ratio": analyze_sharpe,
            "sortino_ratio": analyze_sharpe,
            "calmar_ratio": analyze_sharpe,
            "beta": analyze_sharpe,
            "alpha": analyze_sharpe
        }.get(key, lambda x: "")

        analysis = analysis_func(value)
        metric_display.append({
            "Metric": display_key,
            "Value": display_value,
            "Analysis": analysis
        })

    metrics_df = pd.DataFrame.from_dict(metric_display)
    st.table(metrics_df.style.set_properties(**{
        'text-align': 'left',
        'padding': '5px'
    }))

def compare_portfolios(base_metrics, optimized_metrics, lang):
    """
    Compare base and optimized portfolios and display the comparison table.
    """
    comparison_data = []
    better_portfolio = ""
    better_metric = ""

    for key in base_metrics.keys():
        base_value = base_metrics[key]
        optimized_value = optimized_metrics[key]
        metric_display = get_translated_text(lang, key)

        if key in ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "alpha"]:
            if optimized_value > base_value:
                better = "Optimized"
                better_portfolio = "Optimized"
                better_metric = metric_display
            else:
                better = "Base"
                better_portfolio = "Base"
                better_metric = metric_display
        elif key in ["var", "cvar", "max_drawdown", "beta", "hhi"]:
            if optimized_value < base_value:
                better = "Optimized"
                better_portfolio = "Optimized"
                better_metric = metric_display
            else:
                better = "Base"
                better_portfolio = "Base"
                better_metric = metric_display
        else:
            better = "-"

        def format_val(k, v):
            if k in ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "alpha"]:
                return f"{v:.2f}"
            elif k in ["var", "cvar", "max_drawdown", "beta", "hhi"]:
                return f"{v:.4f}" if k in ["hhi"] else f"{v:.2%}"
            else:
                return f"{v:.2f}"

        comparison_data.append({
            "Metric": metric_display,
            "Base Portfolio": format_val(key, base_value),
            "Optimized Portfolio": format_val(key, optimized_value),
            "Better": better
        })

    comparison_df = pd.DataFrame(comparison_data)

    def highlight_better(row):
        better = row['Better']
        styles = [''] * len(row)
        if better == "Optimized":
            styles[comparison_df.columns.get_loc("Optimized Portfolio")] = 'background-color: lightgreen'
        elif better == "Base":
            styles[comparison_df.columns.get_loc("Base Portfolio")] = 'background-color: lightgreen'
        return styles

    comparison_df = comparison_df.style.apply(highlight_better, axis=1)

    st.markdown("<h3>📊 Comparison: Sharpe vs Base Portfolio</h3>", unsafe_allow_html=True)
    st.table(comparison_df)

    if better_metric:
        recommendation_text = translations[lang].get("recommendation", "").format(
            better_portfolio=better_portfolio, better_metric=better_metric
        )
        st.markdown(f"<p><strong>Recommendation:</strong> {recommendation_text}</p>", unsafe_allow_html=True)


##############################
# Streamlit App
##############################
def main():
    # Language Selection
    st.sidebar.header("🌐 Language Selection")
    selected_language = st.sidebar.selectbox("Select Language:", options=list(languages.keys()), index=0)
    lang = languages[selected_language]

    # Title
    st.title(get_translated_text(lang, "title"))

    # Sidebar for User Inputs
    st.sidebar.header(get_translated_text(lang, "user_inputs"))

    # Define preset universes
    universe_options = {
        'Tech Giants': ['AAPL - Apple', 'MSFT - Microsoft', 'GOOGL - Alphabet', 'AMZN - Amazon', 'META - Meta Platforms', 'TSLA - Tesla', 'NVDA - NVIDIA', 'ADBE - Adobe', 'INTC - Intel', 'CSCO - Cisco'],
        'Finance Leaders': ['JPM - JPMorgan Chase', 'BAC - Bank of America', 'WFC - Wells Fargo', 'C - Citigroup', 'GS - Goldman Sachs', 'MS - Morgan Stanley', 'AXP - American Express', 'BLK - BlackRock', 'SCHW - Charles Schwab', 'USB - U.S. Bancorp'],
        'Healthcare Majors': ['JNJ - Johnson & Johnson', 'PFE - Pfizer', 'UNH - UnitedHealth', 'MRK - Merck', 'ABBV - AbbVie', 'ABT - Abbott', 'TMO - Thermo Fisher Scientific', 'MDT - Medtronic', 'DHR - Danaher', 'BMY - Bristol-Myers Squibb'],
        'Custom': []
    }

    universe_choice = st.sidebar.selectbox(get_translated_text(lang, "select_universe"), options=list(universe_options.keys()), index=0)

    if universe_choice == 'Custom':
        custom_tickers = st.sidebar.text_input(
            get_translated_text(lang, "custom_tickers"),
            value=""
        )
        selected_universe_assets = []
    else:
        selected_universe_assets = st.sidebar.multiselect(
            get_translated_text(lang, "add_portfolio"),
            universe_options[universe_choice],
            default=[]
        )
        custom_tickers = ""

    # Initialize Session State
    if 'my_portfolio' not in st.session_state:
        st.session_state['my_portfolio'] = []
    if 'base_portfolio_metrics' not in st.session_state:
        st.session_state['base_portfolio_metrics'] = None
    if 'optimized_portfolio_metrics' not in st.session_state:
        st.session_state['optimized_portfolio_metrics'] = None

    # Add Assets to Portfolio
    if universe_choice != 'Custom':
        if selected_universe_assets:
            if st.sidebar.button(get_translated_text(lang, "add_portfolio")):
                new_tickers = [extract_ticker(asset) for asset in selected_universe_assets]
                st.session_state['my_portfolio'] = list(set(st.session_state['my_portfolio'] + new_tickers))
                st.sidebar.success(get_translated_text(lang, "add_portfolio"))
    else:
        if st.sidebar.button(get_translated_text(lang, "add_portfolio")):
            if custom_tickers.strip():
                new_tickers = [ticker.strip().upper() for ticker in custom_tickers.split(",") if ticker.strip()]
                st.session_state['my_portfolio'] = list(set(st.session_state['my_portfolio'] + new_tickers))
                st.sidebar.success(get_translated_text(lang, "add_portfolio"))

    # Display Portfolio
    st.sidebar.subheader(get_translated_text(lang, "my_portfolio"))
    if st.session_state['my_portfolio']:
        st.sidebar.write(", ".join(st.session_state['my_portfolio']))
    else:
        st.sidebar.write(get_translated_text(lang, "no_assets"))

    # Optimization Parameters
    st.sidebar.header(get_translated_text(lang, "optimization_parameters"))

    # Date Inputs
    start_date = st.sidebar.date_input(get_translated_text(lang, "start_date"), value=datetime(2024, 1, 1), max_value=datetime.today())
    end_date = st.sidebar.date_input(get_translated_text(lang, "end_date"), value=datetime(2024, 12, 31), max_value=datetime(2030, 12, 31))
    risk_free_rate = st.sidebar.number_input(get_translated_text(lang, "risk_free_rate"), value=2.0, step=0.1) / 100

    investment_strategy = st.sidebar.radio(
        get_translated_text(lang, "investment_strategy"),
        (get_translated_text(lang, "strategy_risk_free"), get_translated_text(lang, "strategy_profit"))
    )

    if investment_strategy == get_translated_text(lang, "strategy_risk_free"):
        specific_target_return = st.sidebar.slider(
            get_translated_text(lang, "target_return"),
            min_value=-5.0, max_value=20.0, value=5.0, step=0.1
        ) / 100
    else:
        specific_target_return = None

    # Advanced optimization parameters
    st.sidebar.header("🔬 Advanced Optimization")
    uncertainty_level = st.sidebar.slider(
        get_translated_text(lang, "uncertainty_level"),
        min_value=5.0, max_value=30.0, value=10.0, step=1.0
    ) / 100

    # NEW: Interval linear system demo controls
    st.sidebar.header("📐 Interval Linear System (n assets demo)")
    interval_k_sigma = st.sidebar.slider(
        "k · σ radius for b̃ intervals",
        min_value=0.5, max_value=3.0, value=1.0, step=0.1
    )
    interval_A_uncertainty = st.sidebar.slider(
        "Relative uncertainty in Ã diagonals (%)",
        min_value=0.0, max_value=20.0, value=5.0, step=1.0
    ) / 100.0
    solve_interval_demo = st.sidebar.button("Solve Interval Linear System (n×n Hansen)")

    # Buttons
    train_lstm = st.sidebar.button(get_translated_text(lang, "train_lstm"))
    optimize_portfolio = st.sidebar.button(get_translated_text(lang, "optimize_portfolio"))
    optimize_sharpe = st.sidebar.button(get_translated_text(lang, "optimize_sharpe"))
    robust_optimize = st.sidebar.button(get_translated_text(lang, "robust_optimize"))
    ilp_optimize = st.sidebar.button(get_translated_text(lang, "ilp_optimize"))
    compare_portfolios_btn = st.sidebar.button(get_translated_text(lang, "compare_portfolios"))

    # Main Area
    st.header(get_translated_text(lang, "portfolio_analysis"))

    # Train LSTM Section
    if train_lstm:
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_lstm"))
        else:
            try:
                clean_tickers = [ticker for ticker in st.session_state['my_portfolio']]
                optimizer = PortfolioOptimizer(clean_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate)
                optimizer.fetch_data()

                X_train, y_train, X_test, y_test, scaler = optimizer.prepare_data_for_lstm()
                model = optimizer.train_lstm_model(X_train, y_train, epochs=10, batch_size=32)
                mae, rmse, r2 = optimizer.evaluate_model(model, scaler, X_test, y_test)

                st.success(get_translated_text(lang, "success_lstm"))

                st.subheader("LSTM Model Evaluation Metrics")
                eval_metrics = {
                    "Mean Absolute Error (MAE)": mae,
                    "Root Mean Squared Error (RMSE)": rmse,
                    "R-squared (R²)": r2
                }
                eval_df = pd.DataFrame.from_dict(eval_metrics, orient='index', columns=['Value'])
                st.table(eval_df.style.format({"Value": "{:.4f}"}))

                # Predict multi-step future returns (average across assets for plotting)
                steps = 30
                future_returns_matrix = optimizer.predict_future_returns(model, scaler, steps=steps)
                # Aggregate to a single portfolio-like series: mean of asset returns per day
                future_returns_series = future_returns_matrix.mean(axis=1)

                future_dates = pd.date_range(end_date, periods=len(future_returns_series) + 1, freq='B')[1:]

                prediction_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Returns (avg across assets)': future_returns_series
                })

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(prediction_df['Date'], prediction_df['Predicted Returns (avg across assets)'],
                        label="Predicted Avg Returns")
                ax.set_xlabel("Date")
                ax.set_ylabel("Predicted Returns")
                ax.set_title(get_translated_text(lang, "train_lstm"))
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

                with st.expander(get_translated_text(lang, "more_info_lstm")):
                    explanation = get_translated_text(lang, "explanation_lstm")
                    st.markdown(explanation)

            except Exception as e:
                logger.exception("An error occurred during LSTM training or prediction.")
                st.error(f"{e}")

    # Standard Optimization
    if optimize_portfolio:
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_opt"))
        elif start_date >= end_date:
            st.error(get_translated_text(lang, "error_date"))
        else:
            try:
                clean_tickers = [ticker for ticker in st.session_state['my_portfolio']]
                optimizer = PortfolioOptimizer(clean_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate)
                updated_tickers = optimizer.fetch_data()

                if investment_strategy == get_translated_text(lang, "strategy_risk_free"):
                    if specific_target_return is None:
                        st.error("Please select a target return for Risk-free Investment strategy.")
                        st.stop()
                    else:
                        optimal_weights = optimizer.min_volatility(specific_target_return)
                else:
                    optimal_weights = optimizer.optimize_sharpe_ratio()

                portfolio_return, portfolio_volatility, sharpe_ratio = optimizer.portfolio_stats(optimal_weights)
                var_95 = optimizer.value_at_risk(optimal_weights, confidence_level=0.95)
                cvar_95 = optimizer.conditional_value_at_risk(optimal_weights, confidence_level=0.95)
                max_dd = optimizer.maximum_drawdown(optimal_weights)
                hhi = optimizer.herfindahl_hirschman_index(optimal_weights)

                allocation = pd.DataFrame({
                    "Asset": updated_tickers,
                    "Weight (%)": np.round(optimal_weights * 100, 2)
                })
                allocation = allocation[allocation['Weight (%)'] > 0].reset_index(drop=True)

                target_display = round(specific_target_return*100, 2) if specific_target_return is not None else "N/A"
                st.subheader(get_translated_text(lang, "allocation_title").format(target=target_display))
                st.dataframe(allocation.style.format({"Weight (%)": "{:.2f}"}))

                metrics = {
                    "var": var_95,
                    "cvar": cvar_95,
                    "max_drawdown": max_dd,
                    "hhi": hhi,
                    "sharpe_ratio": sharpe_ratio,
                    "sortino_ratio": 0.0,
                    "calmar_ratio": 0.0,
                    "beta": 0.0,
                    "alpha": 0.0
                }

                if investment_strategy == get_translated_text(lang, "strategy_risk_free"):
                    st.session_state['base_portfolio_metrics'] = metrics
                else:
                    st.session_state['optimized_portfolio_metrics'] = metrics

                st.subheader(get_translated_text(lang, "performance_metrics"))
                display_metrics_table(metrics, lang)

                st.subheader(get_translated_text(lang, "visual_analysis"))
                col1, col2 = st.columns(2)

                with col1:
                    fig1, ax1 = plt.subplots(figsize=(5, 4))
                    ax1.pie(allocation['Weight (%)'], labels=allocation['Asset'], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
                    ax1.axis('equal')
                    ax1.set_title(get_translated_text(lang, "portfolio_composition"))
                    st.pyplot(fig1)

                with col2:
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    performance_metrics = {
                        "Expected\nAnnual Return (%)": portfolio_return * 100,
                        "Annual Volatility\n(Risk) (%)": portfolio_volatility * 100,
                        "Sharpe Ratio": sharpe_ratio
                    }
                    metrics_bar = pd.DataFrame.from_dict(performance_metrics, orient='index', columns=['Value'])
                    sns.barplot(x=metrics_bar.index, y='Value', data=metrics_bar, palette='viridis', ax=ax2)
                    ax2.set_title(get_translated_text(lang, "portfolio_metrics"))
                    for p in ax2.patches:
                        ax2.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                     ha='center', va='bottom', fontsize=10)
                    plt.xticks(rotation=0, ha='center')
                    plt.tight_layout()
                    st.pyplot(fig2)

                st.subheader(get_translated_text(lang, "correlation_heatmap"))
                correlation_matrix = optimizer.returns.corr()
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', linewidths=0.3, ax=ax3,
                            cbar_kws={'shrink': 0.8}, annot_kws={'fontsize': 8})
                plt.title(get_translated_text(lang, "correlation_heatmap"))
                plt.tight_layout()
                st.pyplot(fig3)

                st.success(get_translated_text(lang, "success_optimize"))

            except Exception as e:
                logger.exception("An unexpected error occurred during optimization.")
                st.error(f"{e}")

    # Sharpe Optimization
    if optimize_sharpe:
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_opt"))
        elif start_date >= end_date:
            st.error(get_translated_text(lang, "error_date"))
        else:
            try:
                clean_tickers = [ticker for ticker in st.session_state['my_portfolio']]
                optimizer = PortfolioOptimizer(clean_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate)
                updated_tickers = optimizer.fetch_data()

                optimal_weights = optimizer.optimize_sharpe_ratio()
                portfolio_return, portfolio_volatility, sharpe_ratio = optimizer.portfolio_stats(optimal_weights)
                var_95 = optimizer.value_at_risk(optimal_weights, confidence_level=0.95)
                cvar_95 = optimizer.conditional_value_at_risk(optimal_weights, confidence_level=0.95)
                max_dd = optimizer.maximum_drawdown(optimal_weights)
                hhi = optimizer.herfindahl_hirschman_index(optimal_weights)

                allocation = pd.DataFrame({
                    "Asset": updated_tickers,
                    "Weight (%)": np.round(optimal_weights * 100, 2)
                })
                allocation = allocation[allocation['Weight (%)'] > 0].reset_index(drop=True)

                st.subheader("🔑 Optimal Portfolio Allocation (Highest Sharpe Ratio)")
                st.dataframe(allocation.style.format({"Weight (%)": "{:.2f}"}))

                metrics = {
                    "var": var_95,
                    "cvar": cvar_95,
                    "max_drawdown": max_dd,
                    "hhi": hhi,
                    "sharpe_ratio": sharpe_ratio,
                    "sortino_ratio": 0.0,
                    "calmar_ratio": 0.0,
                    "beta": 0.0,
                    "alpha": 0.0
                }

                st.session_state['optimized_portfolio_metrics'] = metrics

                st.subheader(get_translated_text(lang, "performance_metrics"))
                display_metrics_table(metrics, lang)

                st.subheader(get_translated_text(lang, "visual_analysis"))
                col1, col2 = st.columns(2)

                with col1:
                    fig1, ax1 = plt.subplots(figsize=(5, 4))
                    ax1.pie(allocation['Weight (%)'], labels=allocation['Asset'], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
                    ax1.axis('equal')
                    ax1.set_title(get_translated_text(lang, "portfolio_composition"))
                    st.pyplot(fig1)

                with col2:
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    performance_metrics = {
                        "Expected\nAnnual Return (%)": portfolio_return * 100,
                        "Annual Volatility\n(Risk) (%)": portfolio_volatility * 100,
                        "Sharpe Ratio": sharpe_ratio
                    }
                    metrics_bar = pd.DataFrame.from_dict(performance_metrics, orient='index', columns=['Value'])
                    sns.barplot(x=metrics_bar.index, y='Value', data=metrics_bar, palette='viridis', ax=ax2)
                    ax2.set_title(get_translated_text(lang, "portfolio_metrics"))
                    for p in ax2.patches:
                        ax2.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                     ha='center', va='bottom', fontsize=10)
                    plt.xticks(rotation=0, ha='center')
                    plt.tight_layout()
                    st.pyplot(fig2)

                st.subheader("📈 Efficient Frontier")
                results, weights_record = optimizer.compute_efficient_frontier()
                portfolio_volatility_arr = results[0]
                portfolio_return_arr = results[1]
                sharpe_ratios = results[2]

                max_sharpe_idx = np.argmax(sharpe_ratios)
                max_sharpe_vol = portfolio_volatility_arr[max_sharpe_idx]
                max_sharpe_ret = portfolio_return_arr[max_sharpe_idx]

                fig4, ax4 = plt.subplots(figsize=(10, 6))
                scatter = ax4.scatter(portfolio_volatility_arr, portfolio_return_arr, c=sharpe_ratios, cmap='viridis', marker='o', s=10, alpha=0.3)
                ax4.scatter(max_sharpe_vol, max_sharpe_ret, c='red', marker='*', s=200, label='Max Sharpe Ratio')
                plt.colorbar(scatter, label='Sharpe Ratio')
                ax4.set_xlabel('Annual Volatility (Risk)')
                ax4.set_ylabel('Expected Annual Return')
                ax4.set_title('Efficient Frontier')
                ax4.legend()
                plt.tight_layout()
                st.pyplot(fig4)

                st.success(get_translated_text(lang, "explanation_sharpe_button"))

            except Exception as e:
                logger.exception("An unexpected error occurred during Sharpe Ratio optimization.")
                st.error(f"{e}")

    # Robust Optimization
    if robust_optimize:
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_opt"))
        elif start_date >= end_date:
            st.error(get_translated_text(lang, "error_date"))
        else:
            try:
                st.subheader(get_translated_text(lang, "robust_optimization"))
                with st.expander("ℹ️ " + get_translated_text(lang, "robust_optimization")):
                    st.markdown(get_translated_text(lang, "explanation_robust"))

                clean_tickers = [ticker for ticker in st.session_state['my_portfolio']]
                optimizer = PortfolioOptimizer(clean_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate)
                updated_tickers = optimizer.fetch_data()

                optimal_weights, scenarios = optimizer.robust_optimization(uncertainty_level)

                allocation = pd.DataFrame({
                    "Asset": updated_tickers,
                    "Weight (%)": np.round(optimal_weights * 100, 2)
                })
                allocation = allocation[allocation['Weight (%)'] > 0].reset_index(drop=True)

                st.subheader(get_translated_text(lang, "robust_results"))
                st.dataframe(allocation.style.format({"Weight (%)": "{:.2f}"}))

                st.subheader(get_translated_text(lang, "scenario_results"))

                scenario_results = []
                for scenario in scenarios:
                    port_return = np.dot(optimal_weights, scenario['mean_returns'])
                    port_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(scenario['cov_matrix'], optimal_weights)))
                    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0

                    scenario_results.append({
                        'Scenario': scenario['name'],
                        'Expected Return (%)': port_return * 100,
                        'Volatility (%)': port_vol * 100,
                        'Sharpe Ratio': sharpe
                    })

                scenario_df = pd.DataFrame(scenario_results)
                st.table(scenario_df.style.format({
                    'Expected Return (%)': '{:.2f}',
                    'Volatility (%)': '{:.2f}',
                    'Sharpe Ratio': '{:.2f}'
                }))

                fig, ax = plt.subplots(figsize=(10, 6))
                scenarios_names = [s['Scenario'] for s in scenario_results]
                returns = [s['Expected Return (%)'] for s in scenario_results]
                volatilities = [s['Volatility (%)'] for s in scenario_results]

                x = np.arange(len(scenarios_names))
                width = 0.35

                ax.bar(x - width/2, returns, width, label='Expected Return (%)', alpha=0.7)
                ax.bar(x + width/2, volatilities, width, label='Volatility (%)', alpha=0.7)

                ax.set_xlabel('Scenarios')
                ax.set_ylabel('Percentage (%)')
                ax.set_title('Robust Portfolio Performance Across Scenarios')
                ax.set_xticks(x)
                ax.set_xticklabels(scenarios_names)
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

                st.success("✅ Robust optimization completed successfully!")

            except Exception as e:
                logger.exception("An error occurred during robust optimization.")
                st.error(f"{e}")

    # ILP Optimization
    if ilp_optimize:
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_opt"))
        elif start_date >= end_date:
            st.error(get_translated_text(lang, "error_date"))
        else:
            try:
                st.subheader(get_translated_text(lang, "ilp_optimization"))
                with st.expander("ℹ️ " + get_translated_text(lang, "ilp_optimization")):
                    st.markdown(get_translated_text(lang, "explanation_ilp"))

                clean_tickers = [ticker for ticker in st.session_state['my_portfolio']]
                optimizer = PortfolioOptimizer(clean_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate)
                updated_tickers = optimizer.fetch_data()

                ilp_result = optimizer.interval_linear_programming(uncertainty_level)

                if ilp_result:
                    st.subheader(get_translated_text(lang, "ilp_results"))

                    st.markdown("### " + get_translated_text(lang, "weight_intervals"))
                    weight_intervals_data = []
                    for i, ticker in enumerate(updated_tickers):
                        min_w, max_w = ilp_result['weight_intervals'][i]
                        weight_intervals_data.append({
                            'Asset': ticker,
                            'Min Weight (%)': round(min_w * 100, 2),
                            'Max Weight (%)': round(max_w * 100, 2),
                            'Range (%)': round((max_w - min_w) * 100, 2)
                        })

                    weight_df = pd.DataFrame(weight_intervals_data)
                    st.dataframe(weight_df.style.format({
                        'Min Weight (%)': '{:.2f}',
                        'Max Weight (%)': '{:.2f}',
                        'Range (%)': '{:.2f}'
                    }))

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### " + get_translated_text(lang, "return_interval"))
                        return_data = {
                            'Metric': [get_translated_text(lang, "best_case"), get_translated_text(lang, "worst_case"), 'Range'],
                            'Return (%)': [
                                round(ilp_result['return_interval'][1] * 100, 2),
                                round(ilp_result['return_interval'][0] * 100, 2),
                                round((ilp_result['return_interval'][1] - ilp_result['return_interval'][0]) * 100, 2)
                            ]
                        }
                        st.table(pd.DataFrame(return_data))

                    with col2:
                        st.markdown("### " + get_translated_text(lang, "risk_interval"))
                        risk_data = {
                            'Metric': [get_translated_text(lang, "best_case"), get_translated_text(lang, "worst_case"), 'Range'],
                            'Risk (%)': [
                                round(ilp_result['risk_interval'][0] * 100, 2),
                                round(ilp_result['risk_interval'][1] * 100, 2),
                                round((ilp_result['risk_interval'][1] - ilp_result['risk_interval'][0]) * 100, 2)
                            ]
                        }
                        st.table(pd.DataFrame(risk_data))

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    assets = [d['Asset'] for d in weight_intervals_data]
                    min_weights = [d['Min Weight (%)'] for d in weight_intervals_data]
                    max_weights = [d['Max Weight (%)'] for d in weight_intervals_data]

                    y_pos = np.arange(len(assets))
                    ax1.barh(y_pos, max_weights, alpha=0.5, label='Max Weight')
                    ax1.barh(y_pos, min_weights, alpha=0.8, label='Min Weight')
                    ax1.set_yticks(y_pos)
                    ax1.set_yticklabels(assets)
                    ax1.set_xlabel('Weight (%)')
                    ax1.set_title('Weight Intervals for Each Asset')
                    ax1.legend()

                    metrics_labels = ['Return', 'Risk']
                    best_case_vals = [
                        ilp_result['return_interval'][1] * 100,
                        ilp_result['risk_interval'][0] * 100
                    ]
                    worst_case_vals = [
                        ilp_result['return_interval'][0] * 100,
                        ilp_result['risk_interval'][1] * 100
                    ]

                    x = np.arange(len(metrics_labels))
                    width = 0.35

                    ax2.bar(x - width/2, best_case_vals, width, label='Best Case', alpha=0.7)
                    ax2.bar(x + width/2, worst_case_vals, width, label='Worst Case', alpha=0.7)
                    ax2.set_xlabel('Metric')
                    ax2.set_ylabel('Percentage (%)')
                    ax2.set_title('Return and Risk Intervals')
                    ax2.set_xticks(x)
                    ax2.set_xticklabels(metrics_labels)
                    ax2.legend()

                    plt.tight_layout()
                    st.pyplot(fig)

                    st.success("✅ ILP optimization completed successfully!")
                else:
                    st.error("ILP optimization failed. Please try different parameters.")

            except Exception as e:
                logger.exception("An error occurred during ILP optimization.")
                st.error(f"{e}")

    # NEW: Interval Linear System (n×n Hansen demo)
    if solve_interval_demo:
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_opt"))
        elif start_date >= end_date:
            st.error(get_translated_text(lang, "error_date"))
        else:
            try:
                st.subheader("📐 Interval Linear System Demo (n×n, Hansen Method)")
                clean_tickers = [ticker for ticker in st.session_state['my_portfolio']]
                optimizer = PortfolioOptimizer(clean_tickers, start_date.strftime('%Y-%m-%d'),
                                               end_date.strftime('%Y-%m-%d'), risk_free_rate)
                updated_tickers = optimizer.fetch_data()

                n = len(updated_tickers)
                if n < 2:
                    st.error("Please select at least 2 assets for the interval system demo.")
                else:
                    # Annual mean returns and volatilities
                    mu_ann = optimizer.mean_returns.values
                    sigma_ann = np.sqrt(np.diag(optimizer.cov_matrix.values))

                    # b̃ intervals from μ ± kσ
                    b_int = IntervalVector([
                        Interval(mu_ann[i], interval_k_sigma * sigma_ann[i])
                        for i in range(n)
                    ])

                    # Ã ≈ I with interval diagonals (1 ± ε), off-diagonals 0
                    eps = interval_A_uncertainty
                    A_rows = []
                    for i in range(n):
                        row = []
                        for j in range(n):
                            if i == j:
                                row.append(Interval.from_bounds(1.0 - eps, 1.0 + eps))
                            else:
                                row.append(Interval(0.0, 0.0))
                        A_rows.append(row)
                    A_int = IntervalMatrix(A_rows)

                    x_sol = IntervalLinearSystemSolver.solve(A_int, b_int)

                    rows = []
                    for i, t in enumerate(updated_tickers):
                        iv = x_sol[i]
                        rows.append({
                            "Variable": f"x_{i+1} ({t})",
                            "μ_x": round(iv.mu, 6),
                            "ρ_x": round(iv.rho, 6),
                            "Lower": round(iv.lower, 6),
                            "Upper": round(iv.upper, 6)
                        })
                    df_int = pd.DataFrame(rows)
                    st.dataframe(df_int)

                    st.markdown(
                        """
                        **Interpretation for your thesis:**  
                        - We form an *interval* linear system **Ã x̃ = b̃** where  
                          - **b̃** encodes annual return uncertainty per asset (μ ± k·σ).  
                          - **Ã** encodes small uncertainty around the identity matrix (1 ± ε on diagonals).  
                        - The Hansen-style solver computes an enclosing interval solution x̃ for **all** consistent systems.  
                        - This demonstrates a **general n×n interval linear system solver** integrated with real financial data.
                        """
                    )

            except Exception as e:
                logger.exception("An error occurred during interval system demo.")
                st.error(f"{e}")

    # Compare Portfolios
    if compare_portfolios_btn:
        if st.session_state['base_portfolio_metrics'] is None or st.session_state['optimized_portfolio_metrics'] is None:
            st.error("Please optimize both the base portfolio and the highest Sharpe Ratio portfolio before comparing.")
        else:
            base_metrics = st.session_state['base_portfolio_metrics']
            optimized_metrics = st.session_state['optimized_portfolio_metrics']
            compare_portfolios(base_metrics, optimized_metrics, lang)


main()
