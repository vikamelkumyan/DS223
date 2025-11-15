"""
Homework 3: Survival Analysis - Customer Churn Prediction
Author: Viktoria Melkumyan
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import (
    AalenJohansenFitter,
    BreslowFlemingHarringtonFitter,
    ExponentialFitter,
    GeneralizedGammaFitter,
    KaplanMeierFitter,
    LogLogisticFitter,
    LogNormalFitter,
    MixtureCureFitter,
    NelsonAalenFitter,
    PiecewiseExponentialFitter,
    SplineFitter,
    WeibullFitter,
    CoxPHFitter,
    LogLogisticAFTFitter,
    LogNormalAFTFitter,
    WeibullAFTFitter
)
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ChurnSurvivalAnalysis:
    """Customer churn survival analysis pipeline"""

    def __init__(self, data_path):
        """Initialize with data and ensure outputs folder exists"""
        self.data = pd.read_csv(data_path)
        self.original_data = self.data.copy()
        self.models = {}
        self.final_model = None
        self.final_data = None
        os.makedirs('outputs', exist_ok=True)

    def prepare_data(self):
        """Prepare data for survival analysis"""
        print("="*70)
        print("DATA PREPARATION")
        print("="*70)
        print(self.data.head(2))

        # Map binary columns
        binary_cols = ['churn', 'retire', 'voice', 'internet', 'forward']
        for col in binary_cols:
            if col in self.data.columns:
                self.data[col] = self.data[col].map({'Yes': 1, 'No': 0}).astype(int)

        # Handle missing numeric values
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].median())

        # One-hot encode categorical variables
        cat_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        self.data_encoded = pd.get_dummies(self.data, columns=cat_cols, drop_first=True, dtype=int)

        print("\n✓ Data preparation complete!")
        return self.data_encoded

    def exploratory_analysis(self):
        """Perform EDA with histograms, boxplots, correlation, and churn rates"""
        print("\n" + "="*70)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*70)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Tenure by churn
        churned = self.data[self.data['churn'] == 1]['tenure']
        retained = self.data[self.data['churn'] == 0]['tenure']
        axes[0, 0].hist([retained, churned], bins=30, alpha=0.7, color=['green', 'red'], label=['Retained','Churned'])
        axes[0, 0].set_xlabel('Tenure (months)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Tenure Distribution by Churn')
        axes[0, 0].legend()

        # Income by churn
        axes[0, 1].boxplot([retained, churned], labels=['Retained','Churned'])
        axes[0, 1].set_title('Tenure Distribution by Churn')
        axes[0, 1].set_ylabel('Tenure')

        # Age by churn
        age_retained = self.data[self.data['churn'] == 0]['age']
        age_churned = self.data[self.data['churn'] == 1]['age']
        axes[0, 2].boxplot([age_retained, age_churned], labels=['Retained','Churned'])
        axes[0, 2].set_title('Age Distribution by Churn')
        axes[0, 2].set_ylabel('Age')

        # Churn rate by custcat
        churn_by_custcat = pd.crosstab(self.original_data['custcat'], self.original_data['churn'], normalize='index') * 100
        if 'Yes' in churn_by_custcat.columns:
            churn_by_custcat['Yes'].plot(kind='bar', ax=axes[1,0], color='coral')
        axes[1,0].set_title('Churn Rate by Customer Category')
        axes[1,0].set_xlabel('Customer Category')
        axes[1,0].set_ylabel('Churn Rate (%)')
        axes[1,0].tick_params(axis='x', rotation=45)

        # Churn rate by region
        churn_by_region = pd.crosstab(self.original_data['region'], self.original_data['churn'], normalize='index')*100
        if 'Yes' in churn_by_region.columns:
            churn_by_region['Yes'].plot(kind='bar', ax=axes[1,1], color='steelblue')
        axes[1,1].set_title('Churn Rate by Region')
        axes[1,1].set_xlabel('Region')
        axes[1,1].set_ylabel('Churn Rate (%)')
        axes[1,1].tick_params(axis='x', rotation=45)

        # Correlation
        numeric_cols = ['tenure', 'age', 'address', 'income', 'churn']
        corr = self.data[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True, ax=axes[1,2])
        axes[1,2].set_title('Feature Correlation Matrix')

        plt.tight_layout()
        plt.savefig('outputs/eda_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ EDA plot saved: outputs/eda_analysis.png")
        plt.close()

    def build_aft_models(self):
        """Fit univariate and regression survival models"""
        print("\n" + "="*70)
        print("BUILDING SURVIVAL MODELS")
        print("="*70)

        breakpoints = list(range(12, 121, 12))
        knot_locations = list(np.quantile(self.data_encoded['tenure'], [0.25,0.5,0.75]))

        univariate_models = {
            'AalenJohansen': AalenJohansenFitter(),
            'BreslowFlemingHarrington': BreslowFlemingHarringtonFitter(),
            'Exponential': ExponentialFitter(),
            'GeneralizedGamma': GeneralizedGammaFitter(),
            'KaplanMeier': KaplanMeierFitter(),
            'LogLogistic': LogLogisticFitter(),
            'LogNormal': LogNormalFitter(),
            'MixtureCure': MixtureCureFitter(base_fitter=WeibullFitter()),
            'NelsonAalen': NelsonAalenFitter(),
            'PiecewiseExponential': PiecewiseExponentialFitter(breakpoints=breakpoints),
            'Spline': SplineFitter(knot_locations=knot_locations),
            'Weibull': WeibullFitter()
        }

        regression_models = {
            'CoxPH': CoxPHFitter(),
            'LogLogisticAFT': LogLogisticAFTFitter(),
            'LogNormalAFT': LogNormalAFTFitter(),
            'WeibullAFT': WeibullAFTFitter()
        }

        results = {}

        for name, model in univariate_models.items():
            try:
                if name == 'AalenJohansen':
                    model.fit(self.data_encoded['tenure'], event_observed=self.data_encoded['churn'], event_of_interest=1)
                else:
                    model.fit(self.data_encoded['tenure'], event_observed=self.data_encoded['churn'])
                results[name] = {'model': model}
                for attr in ['AIC_','BIC_','concordance_index_']:
                    if hasattr(model, attr):
                        results[name][attr.replace('_','')] = getattr(model, attr)
                print(f"  {name} fitted successfully")
            except Exception as e:
                print(f"  Could not fit {name}: {e}")

        for name, model in regression_models.items():
            try:
                model.fit(self.data_encoded, duration_col='tenure', event_col='churn')
                results[name] = {'model': model}
                for attr in ['AIC_','BIC_','concordance_index_','AIC_partial_']:
                    if hasattr(model, attr):
                        results[name][attr.replace('_','')] = getattr(model, attr)
                print(f"  {name} fitted successfully")
            except Exception as e:
                print(f"  Could not fit {name}: {e}")

        self.models = results
        return results

    def compare_models(self):
        """Compare all fitted models and rank them"""
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)

        comparison_data = []
        for name, info in self.models.items():
            comparison_data.append({
                'Model': name,
                'AIC': info.get('AIC', np.nan),
                'BIC': info.get('BIC', np.nan),
                'Concordance': info.get('concordance', np.nan)
            })

        df = pd.DataFrame(comparison_data)
        df['AIC_rank'] = df['AIC'].rank(ascending=True, na_option='bottom', method='min')
        df['Concordance_rank'] = df['Concordance'].rank(ascending=False, na_option='bottom', method='min')
        df['Rank'] = df.apply(lambda r: r['AIC_rank'] if not np.isnan(r['AIC_rank']) else r['Concordance_rank'], axis=1)
        df = df.sort_values('Rank').reset_index(drop=True)
        print(df[['Model','AIC','BIC','Concordance','Rank']])
        best_model = df.iloc[0]['Model']
        print(f"✓ Best model: {best_model}")
        return best_model
    def visualize_survival_curves(self):
        """Visualize survival curves for all models with median profile"""
        print("\n" + "="*70)
        print("GENERATING SURVIVAL CURVES")
        print("="*70)

        fig, ax = plt.subplots(figsize=(12, 7))

        # Time points for prediction
        times = np.linspace(0, self.data['tenure'].max(), 100)

        # Median customer profile
        median_profile = self.data_encoded.median().to_frame().T

        # Color palette (repeats if more than 10 models)
        import itertools
        colors = itertools.cycle([
            '#e74c3c', '#3498db', '#2ecc71', '#f39c12', 
            '#9b59b6', '#1abc9c', '#95a5a6', '#e67e22', 
            '#34495e', '#2c3e50'
        ])

        for name, result in self.models.items():
            model = result['model']

            # Predict survival function (only for models with predict_survival_function)
            if hasattr(model, 'predict_survival_function'):
                try:
                    surv_func = model.predict_survival_function(median_profile, times=times)
                    # Get AIC or AIC_partial if available
                    aic = result.get('AIC', np.nan)
                    if np.isnan(aic) and hasattr(model, 'AIC_partial_'):
                        aic = model.AIC_partial_

                    ax.plot(
                        times, 
                        surv_func.values.flatten(), 
                        label=f"{name}" + (f" (AIC: {aic:.0f})" if not np.isnan(aic) else ""), 
                        linewidth=2.5, 
                        color=next(colors), 
                        alpha=0.8
                    )
                except Exception as e:
                    print(f"  Could not plot {name}: {e}")

        ax.set_xlabel('Tenure (Months)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
        ax.set_title('Survival Curves Comparison: All Models', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig('outputs/survival_curves_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: outputs/survival_curves_comparison.png")
        plt.close()

    def visualize_all_survival_curves(self, horizon_months=120):
        """Plot survival curves of all fitted models on a single graph"""
        print("\n" + "="*70)
        print("VISUALIZING ALL SURVIVAL CURVES")
        print("="*70)

        times = np.arange(1, horizon_months + 1)
        plt.figure(figsize=(12,8))

        for name, info in self.models.items():
            model = info['model']
            try:
                if hasattr(model, 'predict_survival_function'):
                    # Use final_data if regression, else raw tenure
                    data_for_pred = self.final_data if 'AFT' in name or 'Cox' in name else self.data_encoded
                    if 'AalenJohansen' in name:
                        surv_func = model.cumulative_density_
                        plt.step(surv_func.index, 1-surv_func.values, label=name)
                    else:
                        sf = model.predict_survival_function(data_for_pred.iloc[:5], times=times)
                        plt.plot(times, sf.mean(axis=1), label=name)
                else:
                    continue
            except Exception as e:
                print(f"  Could not plot {name}: {e}")

        plt.xlabel('Tenure (months)')
        plt.ylabel('Survival Probability')
        plt.title('Survival Curves for All Models')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('outputs/all_survival_curves.png', dpi=300, bbox_inches='tight')
        print("✓ All survival curves saved: outputs/all_survival_curves.png")
        plt.show()

    def select_final_model(self, best_model_name):
        """Refit best model on significant features"""
        print("\n" + "="*70)
        print("FINAL MODEL SELECTION & FEATURE SELECTION")
        print("="*70)

        if best_model_name not in self.models:
            raise ValueError(f"{best_model_name} not found")

        model_class = self.models[best_model_name]['model'].__class__()
        model_class.fit(self.data_encoded, duration_col='tenure', event_col='churn')

        # Extract significant features
        significant_features = []
        if hasattr(model_class, 'summary') and model_class.summary is not None:
            df_summary = model_class.summary
            if 'p' in df_summary.columns:
                significant_features = df_summary[df_summary['p'] < 0.05].index.get_level_values(1).tolist()
        cols_to_keep = ['tenure','churn'] + [col for col in significant_features if col in self.data_encoded.columns and col not in ['tenure','churn']]

        self.final_data = self.data_encoded[cols_to_keep]
        self.final_model = model_class.__class__()
        self.final_model.fit(self.final_data, duration_col='tenure', event_col='churn')

        print(f"✓ Final model fitted with {len(cols_to_keep)} features")
        return self.final_model

    def calculate_clv(self, spend_share=0.02, margin=0.6, discount_rate=0.01, horizon_months=120):
        """Calculate per-customer CLV"""
        print("\n" + "="*70)
        print("CALCULATING CUSTOMER LIFETIME VALUE (CLV)")
        print("="*70)

        times = np.arange(1, horizon_months+1)
        clv_list = []

        for idx in self.final_data.index:
            customer_features = self.final_data.loc[[idx]]
            surv_prob = self.final_model.predict_survival_function(customer_features, times=times).values.flatten()
            income_k = self.data.at[idx, 'income']
            monthly_rev = (income_k*1000) * spend_share * margin / 12
            discount_factors = (1 + discount_rate) ** (-times)
            clv = np.sum(monthly_rev * surv_prob * discount_factors)
            clv_list.append(clv)

        self.data['CLV'] = clv_list
        return clv_list

    def segment_analysis(self):
        """Analyze CLV by income, age, and customer segment"""
        print("\n" + "="*70)
        print("SEGMENT ANALYSIS")
        print("="*70)

        df = self.data.copy()
        df['income_quartile'] = pd.qcut(df['income'],4,labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
        df['age_group'] = pd.cut(df['age'], bins=[0,30,45,60,100], labels=['18-30','31-45','46-60','60+'])

        clv_custcat = df.groupby('custcat')['CLV'].agg(['count','mean','median','sum'])
        clv_income = df.groupby('income_quartile')['CLV'].agg(['count','mean','median','sum'])
        clv_age = df.groupby('age_group')['CLV'].agg(['count','mean','median','sum'])

        self._visualize_clv_segments(df)
        return clv_custcat, clv_income, clv_age

    def _visualize_clv_segments(self, df):
        """Plot CLV segment visualizations"""
        fig, axes = plt.subplots(2,2, figsize=(14,10))

        # CLV distribution
        axes[0,0].hist(df['CLV'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0,0].axvline(df['CLV'].mean(), color='red', linestyle='--', linewidth=2)
        axes[0,0].set_title('CLV Distribution')

        # CLV by income
        df.groupby('income_quartile')['CLV'].mean().sort_values().plot(kind='barh', ax=axes[0,1], color='coral')
        axes[0,1].set_title('Average CLV by Income Quartile')

        # CLV by age
        df.groupby('age_group')['CLV'].mean().plot(kind='bar', ax=axes[1,0], color='mediumseagreen')
        axes[1,0].set_title('Average CLV by Age Group')

        # CLV by customer category
        df.groupby('custcat')['CLV'].mean().sort_values().plot(kind='barh', ax=axes[1,1], color='mediumpurple')
        axes[1,1].set_title('Average CLV by Customer Category')

        plt.tight_layout()
        plt.savefig('outputs/clv_segment_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ CLV segment plots saved")

    def retention_budget_analysis(self):
        """Estimate retention budget based on 1-year survival probability"""
        print("\n" + "="*70)
        print("RETENTION BUDGET ANALYSIS")
        print("="*70)

        one_year = np.array([12])
        at_risk = []

        for idx in self.final_data.index:
            customer_features = self.final_data.loc[[idx]]
            prob_1yr = self.final_model.predict_survival_function(customer_features, times=one_year).values.flatten()[0]
            if prob_1yr < 0.7:
                at_risk.append({
                    'customer_id': self.data.at[idx,'ID'],
                    'survival_prob_1yr': prob_1yr,
                    'CLV': self.data.at[idx,'CLV'],
                    'tenure': self.data.at[idx,'tenure'],
                    'income': self.data.at[idx,'income'],
                    'custcat': self.data.at[idx,'custcat']
                })

        at_risk_df = pd.DataFrame(at_risk)
        if not at_risk_df.empty:
            total_at_risk_clv = at_risk_df['CLV'].sum()
            retention_budget = total_at_risk_clv * 0.10
            at_risk_df.to_csv('outputs/at_risk_customers.csv', index=False)
            print(f"✓ At-risk customers saved: outputs/at_risk_customers.csv")
            print(f"Recommended retention budget: ${retention_budget:,.2f}")
        else:
            print("No at-risk customers identified.")

        return at_risk_df


def main():
    """Run full analysis pipeline"""
    print("\n" + "="*70)
    print("CUSTOMER CHURN SURVIVAL ANALYSIS")
    print("="*70)

    analysis = ChurnSurvivalAnalysis('telco.csv')
    analysis.prepare_data()
    analysis.exploratory_analysis()
    analysis.build_aft_models()
    best_model = analysis.compare_models()
    analysis.select_final_model(best_model)
    analysis.visualize_survival_curves()
    analysis.calculate_clv()
    analysis.segment_analysis()
    analysis.retention_budget_analysis()

    analysis.data.to_csv('outputs/customer_clv_results.csv', index=False)
    print("✓ Final results saved: outputs/customer_clv_results.csv")


if __name__ == "__main__":
    main()
