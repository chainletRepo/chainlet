import datetime

class Metric:
    def __init__(self):
        self.block_first_used = 0
        self.block_last_used = 0
        self.amount_received = 0
        self.amount_transferred = 0
        self.susp_transactions = 0
        self.mixing_transactions = 0
        self.ordinary_transactions = 0
        self.associated_addresses = 0
        self.first_zone = None
        self.last_zone = None
        self.group_id = 0
        self.heist_metrics = [0.0] * 6

    def get_risk(self):
        risk = "NORMAL"
        if self.susp_transactions > 0 and self.mixing_transactions == 0:
            risk = "3"
        elif self.susp_transactions == 0 and self.mixing_transactions > 0:
            risk = "8"
        elif self.susp_transactions > 0 and self.mixing_transactions > 0:
            risk = "10"
        return risk

    def get_block_first_used(self):
        return self.block_first_used

    def set_block_first_used(self, block_first_used):
        self.block_first_used = block_first_used

    def get_block_last_used(self):
        return self.block_last_used

    def set_block_last_used(self, block_last_used):
        self.block_last_used = block_last_used

    def get_amount_received(self):
        return self.amount_received

    def set_amount_received(self, amount_received):
        self.amount_received = amount_received

    def set_amount_transferred(self, amount_sent):
        self.amount_transferred = amount_sent

    def get_amount_transferred(self):
        return self.amount_transferred

    def get_susp_transactions(self):
        return self.susp_transactions

    def get_ordinary_transactions(self):
        return self.ordinary_transactions

    def get_associated_addresses(self):
        return self.associated_addresses

    def get_first_zone(self):
        return self.first_zone

    def get_last_zone(self):
        return self.last_zone

    def set_susp_transactions(self, susp_transactions):
        self.susp_transactions = susp_transactions

    def get_mixing_transactions(self):
        return self.mixing_transactions

    def set_mixing_transactions(self, mixing_transactions):
        self.mixing_transactions = mixing_transactions

    def add_to_associated_addresses(self, num_associated_addresses):
        self.associated_addresses += num_associated_addresses

    def __str__(self):
        risk = self.get_risk()
        return f"{self.block_first_used}\t{self.first_zone}\t{self.block_last_used}\t{self.last_zone}\t{self.amount_received}\t{self.amount_transferred}\t{self.ordinary_transactions}\t{self.susp_transactions}\t{self.mixing_transactions}\t{self.associated_addresses}\t{risk}"

    def set_first_zoned_date_time(self, zoned_date_time):
        self.first_zone = zoned_date_time

    def set_last_zoned_date_time(self, zoned_date_time):
        self.last_zone = zoned_date_time

    def set_ordinary_transactions(self, ordinary_transactions):
        self.ordinary_transactions = ordinary_transactions

    def add_group_id(self, id):
        self.group_id = id

    def add_heist_features(self, doubles):
        self.heist_metrics = doubles

    def get_group_id(self):
        return self.group_id

    def get_heist_metrics(self):
        return self.heist_metrics
import datetime

class Metric:
    def __init__(self):
        self.block_first_used = 0
        self.block_last_used = 0
        self.amount_received = 0
        self.amount_transferred = 0
        self.susp_transactions = 0
        self.mixing_transactions = 0
        self.ordinary_transactions = 0
        self.associated_addresses = 0
        self.first_zone = None
        self.last_zone = None
        self.group_id = 0
        self.heist_metrics = [0.0] * 6

    def get_risk(self):
        risk = "NORMAL"
        if self.susp_transactions > 0 and self.mixing_transactions == 0:
            risk = "3"
        elif self.susp_transactions == 0 and self.mixing_transactions > 0:
            risk = "8"
        elif self.susp_transactions > 0 and self.mixing_transactions > 0:
            risk = "10"
        return risk

    def get_block_first_used(self):
        return self.block_first_used

    def set_block_first_used(self, block_first_used):
        self.block_first_used = block_first_used

    def get_block_last_used(self):
        return self.block_last_used

    def set_block_last_used(self, block_last_used):
        self.block_last_used = block_last_used

    def get_amount_received(self):
        return self.amount_received

    def set_amount_received(self, amount_received):
        self.amount_received = amount_received

    def set_amount_transferred(self, amount_sent):
        self.amount_transferred = amount_sent

    def get_amount_transferred(self):
        return self.amount_transferred

    def get_susp_transactions(self):
        return self.susp_transactions

    def get_ordinary_transactions(self):
        return self.ordinary_transactions

    def get_associated_addresses(self):
        return self.associated_addresses

    def get_first_zone(self):
        return self.first_zone

    def get_last_zone(self):
        return self.last_zone

    def set_susp_transactions(self, susp_transactions):
        self.susp_transactions = susp_transactions

    def get_mixing_transactions(self):
        return self.mixing_transactions

    def set_mixing_transactions(self, mixing_transactions):
        self.mixing_transactions = mixing_transactions

    def add_to_associated_addresses(self, num_associated_addresses):
        self.associated_addresses += num_associated_addresses

    def __str__(self):
        risk = self.get_risk()
        return f"{self.block_first_used}\t{self.first_zone}\t{self.block_last_used}\t{self.last_zone}\t{self.amount_received}\t{self.amount_transferred}\t{self.ordinary_transactions}\t{self.susp_transactions}\t{self.mixing_transactions}\t{self.associated_addresses}\t{risk}"

    def set_first_zoned_date_time(self, zoned_date_time):
        self.first_zone = zoned_date_time

    def set_last_zoned_date_time(self, zoned_date_time):
        self.last_zone = zoned_date_time

    def set_ordinary_transactions(self, ordinary_transactions):
        self.ordinary_transactions = ordinary_transactions

    def add_group_id(self, id):
        self.group_id = id

    def add_heist_features(self, doubles):
        self.heist_metrics = doubles

    def get_group_id(self):
        return self.group_id

    def get_heist_metrics(self):
        return self.heist_metrics
