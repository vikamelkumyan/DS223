# HW1: Innovation Diffusion Analysis - Matic Robots Matic

**Student Name:** Viktoria Melkumyan
**Course:** DS223: Marketing Analytics  
Date: October 5, 2025  

---

## 1. Innovation Selection

### 2024 Innovation: Matic Robots Matic

Source: TIME Magazine Best Inventions 2024 https://time.com/collection/best-inventions-2024/  
Link:  https://time.com/7094971/matic-robots-matic/ 

---

## 2. Similar Innovation

### iRobot IRobot Roomba (2002)

Link: https://www.irobot.com/en_US/roomba.html
https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.ubuy.co.am%2Fen%2Fproductuk%2F40TJKB7RS-irobot-roomba-j7&psig=AOvVaw1R2PrTUi1sEC5df_ZJAVD-&ust=1759748672891000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCLCgo-b0jJADFQAAAAAdAAAAABAE -IMAGE

Roomba and Matic have the same function: automized floor cleaning with minimal manual effort. They belong to the same product category (robotic vacuum cleaners) and face similar adoption barriers that are high price sensitivity and sceptitism about their cleaning performance. At the same time they offer similar benefits in forms of convenience and social visibility. 

Technologically, Matic is an improved follower of Roomba. While Roomba mainly has random navigation, Matic advances the concept through camera-based mapping, at the same time storing the collected data on the device for data privacy.

This progression mirrors typical innovation–imitation dynamics in the Bass Model, where Matic’s market diffusion is likely to follow Roomba’s established adoption curve but at an accelerated pace. With over two decades of Roomba market data available, it serves as an ideal look-alike innovation for estimating Matic’s diffusion parameters and forecasting its future market impact.

---
## 3. Historical Data

### Link: https://www.statista.com/statistics/731469/irobot-revenue-worldwide/
### Description: Revenue of iRobot worldwide from 2012 to 2023 (in million U.S. dollars)
### Link: https://www.statista.com/statistics/731473/irobot-consumer-robot-shipments-worldwide/ 
### Description: Consumer robot unit shipments of iRobot worldwide from 2014 to 2018 (in millions)
 
Since there was no publicly available full historical data on sales volume, I will use the iRobot's revenue data from 2012 to 2023 and its robot unit shipments from 2014 to 2018. For the missing years in shipment data, I will try to get it using average unit price and total revenue from that year.

---
## 4. Bass Model Parameters Estimation

The Bass model parameters — **coefficient of innovation (p)**, **coefficient of imitation (q)**, and **market potential (M)** — were estimated using worldwide yearly shipments of iRobot vacuums from 2012 to 2023. Yearly shipments were treated as **new adopters** (\( f(t) \)), and the model was implemented in Python to simulate both yearly and cumulative adoption. `curve_fit` from SciPy was used to determine the values of p, q, and M that best fit the data.  

- Coefficient of innovation (p): *[0.0201]*  
- Coefficient of imitation (q): *[0.3720]*  
- Market potential (M): *[55]*  

The fitted model follows the historical shipments closely (**Figure `shipments_bass_fit.png`**) and provides a foundation for predicting future adoption trends.

---
## 5. Prediction of the diffussion of the innovation

---
## 6. Global or country-specific

Robot vacuums as products are not targeting specific countries, instead they are marketed worldwide and are generally reported on global basis. Similarly, in this case, no data was available for robot vacuums shipments' numbers in specific countries. Besides, the idea of the product doesn't rely on interfere with any country's culture, thus may be consider neutral. The only difference may be the countries with little to no tech readiness, making them have maybe a different adoption patterns. Considering those reasons I have decided ot conduct the analysis in a global scope.

Source: Statista
Link: https://www.statista.com/statistics/1392804/smart-vacuums-or-mowing-robots-ownership/

The graph shows that smart vacuums and moving robots ownership rate is similar in different countries, wihtout any drastic differences. Thus, the innovation in that field can be analyzed without concentrating on specific countries and their adoption rates.
---
## 7. Estimation of the number of adopters







