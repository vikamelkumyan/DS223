# HW1: Innovation Diffusion Analysis - Matic Robots Matic

**Student Name:** Viktoria Melkumyan  
**Course:** DS223: Marketing Analytics  
**Date:** October 5, 2025  

---

## 1. Innovation Selection

### 2024 Innovation: Matic Robots Matic

**Source:** TIME Magazine Best Inventions 2024 https://time.com/collection/best-inventions-2024/  
**Link:**  https://time.com/7094971/matic-robots-matic/ 

## 2. Similar Innovation

### iRobot IRobot Roomba (2002)

**Link:** https://www.irobot.com/en_US/roomba.html

Roomba and Matic have the same function: automized floor cleaning with minimal manual effort. They belong to the same product category (robotic vacuum cleaners) and face similar adoption barriers that are high price sensitivity and sceptitism about their cleaning performance. At the same time they offer similar benefits in forms of convenience and social visibility. 

Technologically, Matic is an improved follower of Roomba. While Roomba mainly has random navigation, Matic advances the concept through camera-based mapping, at the same time storing the collected data on the device for data privacy.

This progression mirrors typical innovation–imitation dynamics in the Bass Model, where Matic’s market diffusion is likely to follow Roomba’s established adoption curve but at an accelerated pace. With over two decades of Roomba market data available, it serves as an ideal look-alike innovation for estimating Matic’s diffusion parameters and forecasting its future market impact.

## 3. Historical Data

**Link:** https://www.statista.com/statistics/731469/irobot-revenue-worldwide/  
**Description:** Revenue of iRobot worldwide from 2012 to 2023 (in million U.S. dollars)  
**Link:** https://www.statista.com/statistics/731473irobot-consumer-robot-shipments-worldwide/   
**Description:** Consumer robot unit shipments of iRobot worldwide from 2014 to 2018 (in millions)  
 
Since there was no publicly available full historical data on sales volume, I will use the iRobot's revenue data from 2012 to 2023 and its robot unit shipments from 2014 to 2018. For the missing years in shipment data, I will try to get it using average unit price and total revenue from that year.

![Bass Model](/Users/macbook/Documents/AUA/DS223/HW1/img/shipments_bass_fit.png)

## 4. Bass Model Parameters Estimation

The Bass model parameters — **coefficient of innovation (p)**, **coefficient of imitation (q)**, and **market potential (M)** - were estimated using worldwide yearly shipments of iRobot vacuums from 2012 to 2023. Yearly shipments were treated as **new adopters** (\( f(t) \)), and the model was implemented in Python to simulate both yearly and cumulative adoption. `curve_fit` from SciPy was used to determine the values of p, q, and M that best fit the data.  

- Coefficient of innovation (p): *[0.0201]*  

Only about 2% of the remaining potential market adopts the product spontaneously without influence from existing users, which is quite a low indicator.

- Coefficient of imitation (q): *[0.3720]*  

Roughly 37% of the remaining non-adopters are influenced by the existing adopters each period. This huge difference in q and p is natural and will persist in a lot of cases.

- Market potential (M): *[55]* 
Total number of potential adopters in the market (in millions).

The fitted model follows the historical shipments closely (**Figure `shipments_bass_fit.png`**) and provides a foundation for predicting future adoption trends.


## 5. Prediction of the diffussion of the innovation

![Predicted Adoption Forecast](/Users/macbook/Documents/AUA/DS223/HW1/img/predicted_adoption_forecast.png)

Based on the estimated Bass Model parameters from iRobot Roomba's historical data (p = 0.0201, q = 0.3720, M = 55 million), the forecast reveals Matic's expected adoption trajectory following a characteristic S-shaped curve.  
Key Findings:  
The high imitation coefficient (q = 0.3720) compared to the innovation coefficient (p = 0.0201) indicates that Matic's adoption will be heavily driven by word-of-mouth and social influence. Approximately 37% of remaining potential adopters will be influenced by existing users each period, creating strong network effects.  
Timeline Projections:  
Peak adoption is expected 8-10 years after market introduction, with 50% market penetration (27.5 million units) achieved within 10-12 years and 90% saturation by year 20. This accelerated timeline compared to Roomba's original diffusion reflects increased consumer familiarity with robotic vacuum technology and a mature smart home ecosystem.  
Adoption Drivers:  
The diffusion will transition from innovator-driven adoption (years 1-3) to steep imitation-driven growth (years 4-12), before entering market saturation (years 13+). Matic's enhanced features—camera-based mapping and on-device data storage—address pain points from earlier generations, potentially accelerating adoption among tech-savvy consumers who value both performance and privacy.  

## 6. Global or country-specific

Robot vacuums as products are not targeting specific countries, instead they are marketed worldwide and are generally reported on global basis. Similarly, in this case, no data was available for robot vacuums shipments' numbers in specific countries. Besides, the idea of the product doesn't rely on interfere with any country's culture, thus may be consider neutral. The only difference may be the countries with little to no tech readiness, making them have maybe a different adoption patterns. Considering those reasons I have decided ot conduct the analysis in a global scope.

The graph shows that smart vacuums and moving robots ownership rate is similar in different countries, wihtout any drastic differences. Thus, the innovation in that field can be analyzed without concentrating on specific countries and their adoption rates.

![Robot Ownership by Country](/Users/macbook/Documents/AUA/DS223/HW1/img/robot_ownership_by_country.png)

**Source:** Statista

**Link:** https://www.statista.com/statistics/1392804/smart-vacuums-or-mowing-robots-ownership/

## 7. Estimation of the number of adopters

Market Potential (M = 55 million units): This estimation is based on global households with favorable characteristics for robotic vacuum adoption—middle to upper income, hard floor surfaces, and tech adoption propensity. Given Matic's premium price point (~$1,800), the market is limited to households willing to invest in high-end cleaning automation.
Adoption Forecast by Phase:  
Years 1-3 (Early Adopter Phase):  

Year 1: ~1.1 million adopters
Year 2: ~2.5 million adopters
Year 3: ~4.2 million adopters
Cumulative: 7.8 million (13% of market)

Early adoption driven by affluent tech enthusiasts valuing advanced features.
Years 4-8 (Rapid Growth Phase):

Year 6: ~9.2 million adopters (peak adoption year)
Cumulative by Year 8: 45.7 million (83% of market)

Imitation-driven explosion as satisfied customers become advocates, driving exponential growth through recommendations and social proof.
Years 9-15 (Maturity Phase):

Annual adoption: 1-4 million units/year
Cumulative by Year 10: 52.3 million (95% of market)

Slowing growth as market approaches saturation, primarily late adopters and early replacements.
Years 16+ (Replacement Phase):
Annual adoption stabilizes at 7-8 million units globally, representing replacement demand.

![Forecast Adoption](/Users/macbook/Documents/AUA/DS223/HW1/img/forecast_adoption.png)






