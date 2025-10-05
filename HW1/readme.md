# iRobot Vacuum Cleaner Bass Diffusion Analysis

## Project Overview
This project analyzes the diffusion of iRobot vacuum cleaners using the **Bass diffusion model**. The aim is to estimate the model parameters, forecast future adoption, and visualize the diffusion path both in terms of **yearly new adopters** and **cumulative adoption**. The analysis is conducted at a **global scope** based on available worldwide shipment data.

The project demonstrates:

- Estimation of Bass model parameters (p, q, M)
- Simulation of adoption over time
- Forecasting future adoption
- Visualization of yearly and cumulative adoption
- Justification of analysis scope (global vs country-specific)

---

## Repository Structure
## Project Directory Structure

- **img/**: This directory contains all the images used in the project.
    - `shipments_bass_fit.png`: Comparison of actual yearly shipments with the Bass model fit.  
    - `predicted_adoption_forecast.png`: Predicted yearly and cumulative adoption of iRobot vacuums over 30 years.  
    - `forecast_adoption.png`: Forecasted adoption (yearly and cumulative) for 5 years beyond the historical data.  
    - `robot_ownership_by_country.png`: Robot vacuum ownership rates across different countries.


- **data/**: This directory holds datasets used for analysis.
  - `irobot_shipments_worldwide_2014_2018.csv`: Number of shipments (in millions) iRobot completed for each year within 2014-2018.
  - `revenue_irobot_worldwide_2012_2023.csv`: Total revenue iRobot generated (in million dollars) each year within 2012-2023.
  - `ownership_rate_of_robots_by_country_2025.csv`: Robotic vacuum ownership rates (in percentage) in different countries according to the surveys conducted in 2025.  

- **report/**: This directory contains the project report.
  - `report.pdf`: Final report in PDF format.  
  - `report_source.md`: Source markdown file to generate the report.  

- **/** (root directory): Holds the code files for the project.
  - `script1.py(r)`: Description of what the script does.  
  - `helper_functions.py(r)`: A collection of helper functions used in the project.  


