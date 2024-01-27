import time
import json
import csv

class TimeMeasure:
    def __init__(self):
        self.times = {}
        self.start_times = {}
        self.initial_time = time.perf_counter()
        self.current_prefix = None
        self.tracking = True

    def set_prefix(self, prefix):
        self.current_prefix = prefix
        if not(prefix in self.start_times):
            self.start_times[prefix] = {}
        if not(prefix in self.times):
            self.times[prefix] = {}

    def cur_times(self):
        if self.current_prefix is None:
            return self.times
        else:
            return self.times[self.current_prefix]
    
    def cur_start_times(self):
        if self.current_prefix is None:
            return self.start_times
        else:
            return self.start_times[self.current_prefix]        

    def start_measure(self, name):
        if not(self.tracking):
            return
        t = time.perf_counter()
        self.cur_start_times()[name] = t
    
    def end_measure(self, name):
        if not(self.tracking):
            return
        cur_time = time.perf_counter() - self.cur_start_times()[name]
        if name in self.cur_times():
            self.cur_times()[name].append(cur_time)
        else:
            self.cur_times()[name] = [cur_time]

    def print_times(self):
        print(self.times)

    def all_times(self):
        return self.times

    def save_stats_json(self, file_name):
        with open("results/" + file_name.replace('/', '_') + '.json', 'w') as jsonfile:
            jsonfile.write(json.dumps(self.times))
    
    def get_total_cur_time(self):
        res = 0.0
        for k, t in self.cur_times().items():
            if (k != "others"):
                res+= t[-1]

        return res

    def disable_tracking(self):
        self.tracking = False
    
    def enable_tracking(self):
        self.tracking = True

    # TODO: This should become more generic.
    def add_time_to_last(self, name, cur_time):
        if not(name in self.cur_times()):
            self.cur_times()[name] = []

        self.cur_times()[name].append(float(cur_time)/1000.0 - self.get_total_cur_time())

    # TODO: Needs to be modified to handle prefix.
    def save_stats_csv(self, file_name):
        with open(file_name + '.csv','w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            headers = list(self.times.keys())
            csvwriter.writerow(headers)
            # TODO: Rewrite this into a better way.
            for i in range(len(self.times[headers[0]])):
                row = []
                for item in headers:
                    row.append(self.times[item][i])
                csvwriter.writerow(row)

    def reset_stats(self):
        self.times = {}