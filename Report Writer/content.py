report_="""## Recommendations
### Recommendation 1
- Maintain a consistent sleep schedule to sustain the current sleep efficiency of 91%, ensuring your body continues to receive adequate rest.
- Keep your bedroom environment stable by maintaining the room temperature at around 20.5°C and humidity at 45%, which are currently optimal for your sleep quality.

### Recommendation 2
- Implement strategies to reduce snoring episodes, such as trying positional therapy or using nasal strips, given the 14 snoring episodes totaling 42 minutes with an average intensity of 48 dB.
- Consider consulting with an ENT specialist to explore potential anatomical factors contributing to your snoring, especially since prolonged snoring can impact overall sleep quality.

### Recommendation 3
- Continue monitoring your heart rate during sleep, as your average heart rate of 62 bpm and variability between 54 bpm and 78 bpm are within healthy ranges.
- Keep track of any changes in your Apnea-Hypopnea Index (AHI) and apnea events to ensure they remain below concerning levels, given your current AHI of 3.2 events/hour.

## Points of Concern
- Experiencing 14 snoring episodes totaling 42 minutes, with an average intensity of 48 dB, which may indicate potential airway resistance issues.
- Recorded a prolonged apnea event lasting 18 seconds, which, while not frequent, warrants attention.

## Potential Issues
- Persistent snoring could lead to disrupted sleep architecture and daytime fatigue if not addressed.
- Even though the current AHI is within a low range, the presence of apnea episodes may increase the risk of developing more significant sleep apnea over time.

## Further Action
- Recommend scheduling a follow-up consultation with a sleep specialist to thoroughly evaluate snoring patterns and apnea events, ensuring early intervention if sleep-disordered breathing progresses.
"""
data_="""
## Sleep Data
Date: August 11, 2024
### Sleep Duration
- Total Sleep Time: 7 hours 23 minutes
- Time in Bed: 8 hours 5 minutes
- Sleep Efficiency: 91%
### Sleep Stages
- Light Sleep: 3 hours 42 minutes (50.2%)
- Deep Sleep: 1 hour 51 minutes (25.1%)
- REM Sleep: 1 hour 50 minutes (24.7%)
### Snoring
- Total Snoring Time: 42 minutes
- Snoring Episodes: 14
- Average Snoring Intensity: 48 dB
### Sleep Apnea
- Apnea-Hypopnea Index (AHI): 3.2 events/hour
- Total Apnea Events: 24
- Longest Apnea Duration: 18 seconds
### Heart Rate
- Average Heart Rate: 62 bpm
- Lowest Heart Rate: 54 bpm
- Highest Heart Rate: 78 bpm
### Other Metrics
- Room Temperature: 20.5°C
- Room Humidity: 45%
- Noise Level: Average 32 dB, Peak 58 dB"""

template1_ = """# Sleep Analysis Report

## Summary
[Provide a concise overview of the patient's sleep patterns, highlighting key observations and potential issues. Use medical terminology appropriately but ensure it's understandable to the patient.]

## Diagnosis
[Offer a specific, medically sound diagnosis based on the sleep data. Include the severity level if applicable. Use medical terminology. Quote specific values from the data.]

## Data Analysis
[Analyze the sleep data, noting any anomalies or patterns. Use bullet points for clarity. Discuss implications of anomalies and patterns.]

- Key Observation 1
- Key Observation 2
- Anomaly 1 (if any)
- Anomaly 2 (if any)

## Recommendations
[Provide at least three actionable, realistic recommendations. Each should be clearly explained and justified.]

### Recommendation 1: [Brief Title]
- Detailed explanation of the recommendation
- Expected benefits
- How to implement this recommendation

### Recommendation 2: [Brief Title]
- Detailed explanation of the recommendation
- Expected benefits
- How to implement this recommendation

### Recommendation 3: [Brief Title]
- Detailed explanation of the recommendation
- Expected benefits
- How to implement this recommendation

## Points of Concern
[List any specific concerns based on the sleep data. Explain why each is concerning and its potential impact on health. Use medical terminology and refer to anomalies in data.]
- Concern 1
- Concern 2

## Potential Health Issues
[Discuss any potential health issues that may be indicated by the sleep data. Use medical terminology but provide clear explanations.]
- Potential Issue 1
- Potential Issue 2

## Further Action
[Provide clear guidance on next steps, including when to seek further medical attention.]
- Immediate actions the patient should take
- Follow-up appointment recommendation (if necessary)
- Referral to a specialist (if required, specify the type of specialist and why)

## Closing Notes
[Summarize the key points and encourage the patient to follow the recommendations and seek further clarification if needed.]
"""


template2_ = """## Recommendations
### Recommendation 1
- Bullet A
- Bullet B
### Recommendation 2
- Bullet A
- Bullet B
### Recommendation 3
- Bullet A
- Bullet B
## Points of Concern
- Mention anomalies in data here
## Potential Issues
- Mention potential health concerns here
## Further Action
- Recommend the user to an expert if required"""

template_ = template2_