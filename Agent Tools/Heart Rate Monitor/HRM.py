class HRM_Model():
    def __init__(self):
        super(HRM_Model, self).__init__()
        self.need_data = True
        self.mean_rhr = 0
        self.stddev_rhr = 0
        self.n_stddevs = 3

    def set_data(self, mean_rhr, stddev_rhr):
        self.need_data = False
        self.mean_rhr = mean_rhr
        self.stddev_rhr = stddev_rhr

    # Checks for anomalous heart rate using basic statstical analysis
    def check_anomaly(self, hr):
        if self.need_data:
            raise Exception("No Data Given to HRH_Model.")
        if self.need_data:
            return False
        if hr < self.mean_rhr - self.n_stddevs * self.stddev_rhr or hr > self.mean_rhr + self.n_stddevs * self.stddev_rhr:
            return True
        return False
    
    # Fail Safe in case statistical analysis fails
    def check_fatal_conditions(self, hr):
        if hr < 30:
            return True
            
    # Agent Wrapper for check_anomaly()
    # Agents need str output for understanding
    def agent_check_anomaly(self, hr):
        if self.check_fatal_conditions(hr):
            return "Heart Rate Critically Low: Heart Rate is below 30 bpm."
        if not self.check_anomaly(hr):
            return "Heart Rate Normal"
        
        anomaly = hr - self.mean_rhr
        n_stdev = anomaly / self.stddev_rhr

        if n_stdev <= -self.n_stddevs:
            return f"Heart Rate Too Slow: {-n_stdev} standard deviations ({-anomaly} bpm) less than normal."
        elif n_stdev >= self.n_stddevs:
            return f"Heart Rate Too Fast: {n_stdev} standard deviations ({anomaly} bpm) more than normal."
        else:
            raise Exception("Failed Anomaly Detection. Error in code.")
             
if __name__ == "__main__":
    model = HRM_Model()
    model.set_data(40, 5)
    print(model.check_anomaly(53)) # False
    print(model.check_anomaly(62)) # True

    print(model.agent_check_anomaly(53)) # Heart Rate Normal
    print(model.agent_check_anomaly(62)) # Heart Rate Too Fast: 4.4 standard deviations (22 bpm) more than normal.
    print(model.agent_check_anomaly(10)) # Heart Rate Too Slow: 6.0 standard deviations (30 bpm) less than normal.
    
    # model2 = HRM_Model()
    # print(model2.check_anomaly(54)) # Error

    