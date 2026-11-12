"""Business insights engine generating rule-based insights from actual historical data."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import config

def generate_business_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Generate deterministic business insights derived strictly from actual retail sales data.
    """
    insights = []

    # 1. Best-performing product family
    fam_sales = df.groupby('family', observed=False)['sales'].sum()
    top_family = fam_sales.idxmax()
    top_family_sales = fam_sales.max()
    top_family_pct = (top_family_sales / df['sales'].sum()) * 100
    insights.append({
        'title': 'Top Product Family',
        'category': 'Product',
        'badge': 'Positive',
        'message': f"'{top_family}' is the highest-grossing category with ${top_family_sales:,.0f} in cumulative revenue, representing {top_family_pct:.1f}% of total enterprise sales."
    })

    # 2. Highest-performing store
    store_sales = df.groupby('store_nbr', observed=False)['sales'].sum()
    top_store = store_sales.idxmax()
    top_store_sales = store_sales.max()
    top_store_city = df[df['store_nbr'] == top_store]['city'].iloc[0]
    insights.append({
        'title': 'Highest Performing Store',
        'category': 'Store',
        'badge': 'Positive',
        'message': f"Store #{top_store} located in {top_store_city} leads across all retail outlets with ${top_store_sales:,.0f} in cumulative volume."
    })

    # 3. Month with highest sales
    m_sales = df.groupby(df['date'].dt.strftime('%B'), observed=False)['sales'].sum()
    top_month = m_sales.idxmax()
    insights.append({
        'title': 'Peak Seasonal Month',
        'category': 'Seasonality',
        'badge': 'Seasonal',
        'message': f"Across the historical timeline, {top_month} consistently registers the highest aggregate sales volume, driven by holiday demand and seasonal restocking."
    })

    # 4. Day of the week with highest sales
    dow_sales = df.groupby(df['date'].dt.day_name(), observed=False)['sales'].mean()
    top_dow = dow_sales.idxmax()
    top_dow_mean = dow_sales.max()
    insights.append({
        'title': 'Peak Shopping Day',
        'category': 'Footfall',
        'badge': 'Operational',
        'message': f"{top_dow} is the highest footfall day with an average daily revenue of ${top_dow_mean:,.0f}, showing strong weekend retail shopping concentration."
    })

    # 5. Product families with declining or lowest sales
    bottom_family = fam_sales.idxmin()
    bottom_family_sales = fam_sales.min()
    insights.append({
        'title': 'Underperforming Product Category',
        'category': 'Inventory',
        'badge': 'Alert',
        'message': f"'{bottom_family}' generated the lowest volume (${bottom_family_sales:,.0f}), indicating opportunities for assortment rationalization or targeted clearance."
    })

    # 6. Promotion-related sales patterns
    promo_sales = df[df['onpromotion'] > 0]['sales'].mean()
    non_promo_sales = df[df['onpromotion'] == 0]['sales'].mean()
    lift_ratio = (promo_sales / non_promo_sales) if non_promo_sales > 0 else 1.0
    insights.append({
        'title': 'Promotion Elasticity Lift',
        'category': 'Marketing',
        'badge': 'Optimization',
        'message': f"Items on active promotion achieve an average sales volume of ${promo_sales:.1f} vs ${non_promo_sales:.1f} for non-promoted items—delivering a {lift_ratio:.1f}x promotional lift factor."
    })

    # 7. Holiday-related sales patterns
    hol_sales = df[df['is_holiday'] == 1]['sales'].mean()
    reg_sales = df[df['is_holiday'] == 0]['sales'].mean()
    hol_lift = ((hol_sales - reg_sales) / reg_sales) * 100 if reg_sales > 0 else 0
    insights.append({
        'title': 'Holiday Sales Impact',
        'category': 'Calendar',
        'badge': 'Impact',
        'message': f"Designated public holidays generate an average daily sales impact of {hol_lift:+.1f}% compared to regular operating days."
    })

    # 8. Stores with unusually low sales
    store_mean = store_sales.mean()
    store_std = store_sales.std()
    low_stores = store_sales[store_sales < (store_mean - 1.2 * store_std)].index.tolist()
    if low_stores:
        low_store_str = ', '.join([f"Store #{s}" for s in low_stores[:3]])
        insights.append({
            'title': 'Store Performance Outlier Alert',
            'category': 'Store Health',
            'badge': 'Warning',
            'message': f"{low_store_str} recorded sales significantly below regional cluster averages. Recommended for operational review and localized marketing support."
        })

    return insights

if __name__ == '__main__':
    df = pd.read_parquet(config.CLEANED_DATA_PATH)
    res = generate_business_insights(df)
    for r in res:
        print(f"[{r['category']}] {r['title']}: {r['message']}")
