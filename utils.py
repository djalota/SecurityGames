#Mapping from Permit Type to Permit Charge ($/month)
permit_dict = {'A PERMIT': 133, 'C PERMIT': 38, 'RESIDENT': 45, 'RESIDENT/C': 41.5, 
               'OTHER PERMIT': 448, 'VISITOR / TIMED SPACE': 713.6}

permit_types = ['A PERMIT', 'C PERMIT', 'RESIDENT', 'RESIDENT/C', 'OTHER PERMIT', 'VISITOR / TIMED SPACE']

threshold_prob = {133: 133/(45*20), 38: 38/(45*20), 45: 45/(45*30), 
                  41.5: 41.5/(41.5*25), 448: 448/(45*20), 713.6: 713.6/(45*20)}