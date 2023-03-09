import os
from IntPair import IntPair
from Node import Node
from Util import Util
import random
import sys
import csv
import datetime

class ArticleOrbit:

    @staticmethod
    def get_bheist_dates(heist_file):
        try:
            path = os.path.abspath(heist_file)
            file = open(path, "r")
            reader = csv.reader(file)
            file_lines = []

            for line in reader:
                file_lines.append(line)

            file.close()
            
            dates = {}

            for line in file_lines:
                if line == file_lines[0]:
                    continue

                arr = line.split(",")
                add = Node(arr[0])
                add.set_label(arr[9].strip())
                year = int(arr[1])
                day_of_the_year = int(arr[2])
                p = IntPair(year, day_of_the_year)

                if p not in dates:
                    dates[p] = set(add)

            return dates

        except Exception as e:
            print(e)

    @staticmethod
    def darknet_market_adds(market_file):
        try:
            path = os.path.abspath(market_file)
            file = open(path, "r")
            reader = csv.reader(file)
            file_lines = []

            for line in reader:
                file_lines.append(line)

            file.close()
            
            adds = {}
            length = len(file_lines[0].split(","))
            label = "darknet"
            
            for l in range(1, len(file_lines)):
                add = file_lines[l].strip()
                adds[add] = label

            return adds
        
        except Exception as e:
            print(e)

    @staticmethod
    def get_ransom_adds(heist_file):
        try:
            path = os.path.abspath(heist_file)
            file = open(path, "r")
            reader = csv.reader(file)
            file_lines = []
            
            for line in reader:
                file_lines.append(line)
                
            file.close()
            
            adds = {}
            length = len(file_lines[0].split(","))

            for l in range(1, len(file_lines)):
                arr = file_lines[l].strip().split(",")
                add = arr[0]
                if arr[length - 1].lower() != "white":
                    adds[add] = arr[9]

            return adds
        
        except Exception as e:
            print(e)

    def main():
        args = sys.argv
        heist_file = args[1]
        orbit_file = args[2]
        extended_orbit_file = "v2" + orbit_file
        dir = args[3]
        darknet_file = args[4]
        huang_file = args[5]
        ArticleOrbit.extract_heist_dates_only(heist_file, orbit_file, dir)
        adds = ArticleOrbit.darknet_market_adds(darknet_file)
        print(len(adds), " market addresses")
        huang_adds = ArticleOrbit.huang_adds(huang_file)
        s1 = set(adds.keys())
        s1 = s1.intersection(set(huang_adds.keys()))
        print(len(s1), " common adds between market and Huang")
        adds = adds.update(huang_adds)
        print(len(adds), " huang+darknet market addresses")
        heist_adds = ArticleOrbit.get_ransom_adds(heist_file)
        s2 = set(heist_adds.keys())
        s2 = s2.intersection(set(adds.keys()))
        print(len(s2), " common adds between (market + Huang) and heist")
        adds = adds.update(heist_adds)
        print(len(adds), " ransom+darknet+Huang's addresses")

        try:
            file = open(extended_orbit_file, "w")
            writer = csv.writer(file)
            writer.writerow(Util.create_orbit_header()+"\r\n")
            sample_rate = 0.005

            for date in Util.get_dates_between("01-01-2010", "01-01-2019"):
                day_of_month = date.strftime("%d")
                day_of_year = date.strftime("%j")
                month = date.month
                year = date.year
                m = "0" + str(month) if month < 10 else str(month)
                d = "0" + str(day_of_month) if day_of_month < 10 else str(day_of_month)
                file_name = dir + str(year) + "_" + m + "_" + d + "address_orbits.csv"
                address_orbits = ArticleOrbit.read_day_orbits(sample_rate, file_name, adds)
                print(str(year) + "\t" + str(month) + "\t" + str(day_of_month) + ", adds of interest were " + str(address_orbits.size()) + ".")

                for k in address_orbits.keys():
                    if k in adds:
                        label = adds[k]
                    else:
                        label = "white"

                    file.write(k + "\t" + str(day_of_year) + "\t" + str(year) + "\t" + label + address_orbits[k] + "\r\n")
                    print(k + "\t" + str(day_of_year) + "\t" + str(year) + "\t" + label)
                
            file.close()

        except Exception as e:
            print(e)


    @staticmethod
    def huang_adds(huang_file):
        try:
            path = os.path.abspath(huang_file)
            file = open(path, "r")
            reader = csv.reader()
            file_lines = []
            
            for line in reader:
                file_lines.append(line)
                
            file.close()
            
            adds = {}

            for l in range(1, len(file_lines)):
                arr = file_lines[l].strip().split(",")
                add = arr[0]
                adds[add] = arr[1]

            return adds

        except Exception as e:
            print(e)

    @staticmethod
    def extract_heist_dates_only(heist_file, orbit_file, dir):
        dates = ArticleOrbit.get_bheist_dates(heist_file)
        cal = datetime.datetime.now()

        try:
            file = open(orbit_file, "w")
            writer = csv.writer(file)
            writer.writerow(Util.create_orbit_header() + "\r\n")

            for ip in dates.keys():
                year = ip.get_left()
                day_of_year = ip.get_right()
                cal = cal.replace(year=year, month=1, day=1)
                cal = cal + datetime.timedelta(days=day_of_year)
                month = cal.month
                day_of_month = cal.day
                addsof_interest = {}

                for n in dates["ip"]:
                    addsof_interest[n.get_hash_id()] = n.get_label()

                do_not_sample_whites = 0
                m = "0" + str(month) if month < 10 else str(month)
                d = "0" + str(day_of_month) if day_of_month < 10 else str(day_of_month)
                file_name = dir + str(year) + "_" + m + "_" + d + "address_orbits.csv"
                address_orbits = ArticleOrbit.read_day_orbits(do_not_sample_whites, file_name, addsof_interest)
                for k in address_orbits.keys():
                    if k in addsof_interest:
                        label = addsof_interest[k]
                    else:
                        label = "white"

                    writer.writerow(k + "\t" + str(day_of_year) + "\t" + str(year) + "\t" + label + address_orbits[k] + "\r\n")
                    
            file.close()

        except Exception as e:
            print(e)

    @staticmethod
    def read_day_orbits(sample_rate, file_name, addresses):
        orbit_buffer = {}
        path = os.path.abspath(file_name)

        if not os.path.exists(path):
            print(file_name + " does not exist.")
            return orbit_buffer

        file = open(file_name, "r")
        reader = csv.reader(file)
        lines = []

        for line in reader:
            lines.append(line)

        file.close()

        bound = 100000
        actual_val = int(bound * sample_rate);
        
        if len(lines) != 0 and lines[0].split("\t")[0].lower() != "address":
            raise Exception("Header is missing in " + file_name)

        for line in lines:
            split = line.split("\t")
            add = split[0]
            ran_val = random.randint(bound)

            if add in addresses or ran_val < actual_val:
                ob = []

                for j in range(1, len(split)):
                    ob.append("\t" + split[j])
                    
                orbit_buffer[add] = "".join(ob)

        return orbit_buffer