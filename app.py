import ast
import json
import random
import re
import string
from pathlib import Path
import numpy as np
import pandas as pd

import pandas as pd
from dateutil.parser import parse
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

from gaCentile import getCentile
#from gaCentile import getCentile, getSFHData, get_plot
#from getQR import genQR, genJSONout

app = Flask(__name__, static_url_path='/static')
api = Api(app)

TODOS = {
    'todo1': {'task': 'build an API'},
    'todo2': {'task': '?????'},
    'todo3': {'task': 'profit!'},
}


def abort_if_todo_doesnt_exist(todo_id):
    if todo_id not in TODOS:
        abort(404, message="Todo {} doesn't exist".format(todo_id))


parser = reqparse.RequestParser()
parser.add_argument('task')


# Todo
# shows a single todo item and lets you delete a todo item
class Todo(Resource):
    def get(self, todo_id):
        abort_if_todo_doesnt_exist(todo_id)
        return TODOS[todo_id]

    def delete(self, todo_id):
        abort_if_todo_doesnt_exist(todo_id)
        del TODOS[todo_id]
        return '', 204

    def put(self, todo_id):
        args = parser.parse_args()
        task = {'task': args['task']}
        TODOS[todo_id] = task
        return task, 201


# TodoList
# shows a list of all todos, and lets you POST to add new tasks
class TodoList(Resource):
    def get(self):
        return TODOS

    def post(self):
        args = parser.parse_args()
        todo_id = int(max(TODOS.keys()).lstrip('todo')) + 1
        todo_id = 'todo%i' % todo_id
        TODOS[todo_id] = {'task': args['task']}
        return TODOS[todo_id], 201


class GACentile(Resource):
    def get(self):
        return 'Submit data'

    def post(self):
        args = parser.parse_args()
        centstring = args['task']
        gainput = re.search('ga(.*)bweight', centstring)
        gainput = int(gainput.group(1))
        bweight = re.search('bweight(.*)gender', centstring)
        bweight = int(bweight.group(1))
        gender = int(centstring.partition('gender')[2])

        uk90m = None
        t = 'loaded'

        filename_data_uk90M = str(Path.cwd() / 'static/uk90M.csv')
        filename_data_uk90F = str(Path.cwd() / 'static/uk90F.csv')
        filename_data_ukwhoM = str(Path.cwd() / 'static/ukwhoM.csv')
        filename_data_ukwhoF = str(Path.cwd() / 'static/ukwhoF.csv')

        centile = 0

        try:
            bw, gastr, centile, cent_ref = getCentile(gainput, bweight, gender)
            t = cent_ref
        except:
            t = 'wont process centile'

        # return t

        test1 = [ele for ele in ['1', '21', '31', '41', '51', '61', '71', '81', '91'] if (ele == str(centile))]
        test2 = [ele for ele in ['2', '22', '32', '42', '52', '62', '72', '82', '92'] if (ele == str(centile))]
        test3 = [ele for ele in ['3', '23', '33', '43', '53', '63', '73', '83', '93'] if (ele == str(centile))]

        if centile == 0:
            centilestr = '0th'
        elif centile == 1:
            centilestr = '%sst' % str(centile)
        elif test1:
            centilestr = '%sst' % str(centile)
        elif test2:
            centilestr = '%snd' % str(centile)
        elif test3:
            centilestr = '%srd' % str(centile)
        else:
            centilestr = '%sth' % str(centile)

        ga_string = {'bweight': bw, 'gastr': gastr, 'centile': centilestr, 'cent_ref': cent_ref}
        ga_string = json.dumps(ga_string)
        ga_string = json.loads(ga_string)

        return ga_string

#
# class GetUID(Resource):
#     def get(self):
#         return 'Submit data'
#
#     def post(self):
#         args = parser.parse_args()
#         uidstring = args['task']
#         nhsinput = re.search('nhs(.*)edd', uidstring)
#         nhsinput = str(nhsinput.group(1))
#         eddinput = re.search('edd(.*)dob', uidstring)
#         eddinput = str(eddinput.group(1))
#         dobinput = str(uidstring.partition('dob')[2])
#         uid = ''.join(random.sample(string.ascii_lowercase, 6))
#         filename_registration = 'static/register/%s%s%s.txt' % (nhsinput, dobinput, eddinput)
#         filename_registration = str(Path.cwd() / filename_registration)
#         path1 = Path(filename_registration)
#         filename_data = 'static/%s%s.json' % (uid, eddinput)
#         filename_data = str(Path.cwd() / '%s') % filename_data
#         path2 = Path(filename_data)
#
#         inputlist = [uid]
#
#         if path1.is_file() is False:
#             with open(filename_registration, 'w') as f:
#                 f.writelines(inputlist)
#                 eddouput = '%s/%s' % (eddinput[:2], eddinput[2:])
#                 uid_str = {"uid_str": uid, "edd_str": eddouput, "nhs_str": nhsinput}
#                 uid_str = json.dumps(uid_str)
#                 uid_str = json.loads(uid_str)
#                 return uid_str
#
#         elif path1.is_file() is True:
#             with open(filename_registration) as f:
#                 eddouput = '%s/%s' % (eddinput[:2], eddinput[2:])
#                 contents = f.readlines()
#                 uid_str = {"uid_str": contents[0], "edd_str": eddouput, "nhs_str": nhsinput}
#                 uid_str = json.dumps(uid_str)
#                 uid_str = json.loads(uid_str)
#                 return uid_str
#
#
# class GetQR(Resource):
#     def get(self):
#         return 'Submit data'
#
#     def post(self):
#         args = parser.parse_args()
#         # up_str = {'banana' : 'chips'}
#         uploadstring = args['task']
#         json_preg = genJSONout(uploadstring)
#         qrimageb64 = genQR(json_preg)
#         # qrconfirm = "CodeAccepted"
#         # up_str = {"dataout", json_preg}
#         uploadstring = json.dumps(uploadstring)
#         uploadstring = json.loads(uploadstring)
#         return uploadstring
#
#
# class StoreSFH(Resource):
#     def get(self):
#         return 'Submit data'
#
#     def post(self):
#         args = parser.parse_args()
#         uploadstring = args['task']
#         eddinput = re.search('edd(.*)nhs', uploadstring)
#         eddinput = str(eddinput.group(1))
#         nhsinput = re.search('nhs(.*)uid', uploadstring)
#         nhsinput = str(nhsinput.group(1))
#         uidinput = re.search('uid(.*)date', uploadstring)
#         uidinput = str(uidinput.group(1))
#         dateinput = re.search('date(.*)sfh', uploadstring)
#         dateinput = str(dateinput.group(1))
#         sfhinput = re.search('sfh(.*)user', uploadstring)
#         sfhinput = str(sfhinput.group(1))
#         userinput = re.search('user(.*)ga_int', uploadstring)
#         userinput = str(userinput.group(1))
#         ga_intinput = re.search('ga_int(.*)ga_str', uploadstring)
#         ga_intinput = str(ga_intinput.group(1))
#         ga_strinput = str(uploadstring.partition('ga_str')[2])
#         rowinput = str(uploadstring.partition('row')[2])
#         testpath = 'static/data/%s%s.json' % (uidinput, eddinput)
#         testpath = str(Path.cwd() / '%s') % testpath
#         path = Path(testpath)
#
#         if path.is_file() is False:
#             my_json = {"sfhdata": []}
#             dateoutput = '%s/%s' % (dateinput[:2], dateinput[2:])
#             my_json['sfhdata'].append(
#                 {"datestr": dateoutput, "sfh": sfhinput, "user": userinput, "ga_int": ga_intinput,
#                  "ga_str": ga_strinput,
#                  "row": rowinput})
#             json_string = json.dumps(my_json)
#
#             with open(path, 'w') as outfile:
#                 json.dump(json_string, outfile)
#
#             return "New file created and saved"
#
#         if path.is_file() is True:
#             with open(path) as json_file:
#                 my_json = json.load(json_file)
#                 my_json = json.loads(my_json)
#                 dateoutput = '%s/%s' % (dateinput[:2], dateinput[2:])
#                 my_json['sfhdata'].append(
#                     {"datestr": dateoutput, "sfh": sfhinput, "user": userinput, "ga_int": ga_intinput,
#                      "ga_str": ga_strinput, "row": rowinput})
#                 json_string = json.dumps(my_json)
#
#                 with open(path, 'w') as outfile:
#                     json.dump(json_string, outfile)
#
#                 return "Saved by appending to existing file"
#
#
# class RetrieveSFH(Resource):
#     def get(self):
#         return 'Submit data'
#
#     def post(self):
#         args = parser.parse_args()
#         downloadstring = args['task']
#         eddinput = re.search('edd(.*)nhs', downloadstring)
#         eddinput = str(eddinput.group(1))
#         nhsinput = re.search('nhs(.*)uid', downloadstring)
#         nhsinput = str(nhsinput.group(1))
#         uidinput = str(downloadstring.partition('uid')[2])
#         testpath = 'static/data/%s%s.json' % (uidinput, eddinput)
#         testpath = str(Path.cwd() / '%s') % testpath
#         path = Path(testpath)
#
#         if path.is_file() is False:
#             return 'No data stored'
#         elif path.is_file() is True:
#             with open(path) as json_file:
#                 data = json.load(json_file)
#                 data = json.loads(data)
#                 return data
#
#
# class GetSFH(Resource):
#     def get(self):
#         return 'Submit SFH data'
#
#     def post(self):
#         args = parser.parse_args()
#         sfhstring = args['task']
#         sfhcase = []
#
#         filename_sfh = str(Path.cwd() / 'static/sfhdata.csv')
#         sfhdata = pd.read_csv(filename_sfh)
#         uploadstring = str(sfhstring)
#
#         startidx = [idx for idx, item in enumerate(uploadstring) if '{' in item]
#         endidx = [idx for idx, item in enumerate(uploadstring) if '}' in item]
#         sfhcount = len(startidx)
#
#         sfhcase = pd.DataFrame(index=range(sfhcount), columns=range(10))
#         sfhcase.columns = ['idx', 'ID', 'SFH', 'SFHDate', 'SFHStaff', 'EDD', 'SFHStr', 'GA', 'GAdw', 'Centile']
#
#         for d in range(sfhcount):
#             temp = uploadstring[startidx[d]: endidx[d] + 1]
#             try:
#                 tempjson = ast.literal_eval(temp)
#             except:
#                 continue
#
#             tempga = (280 - (parse(tempjson['edd']).date() - parse(tempjson['sfhdate']).date()).days) / 7
#             tempgastring, tempgadw, tempsfhcentile = getSFHData(tempga, tempjson['sfh'], sfhdata)
#             sfhcase.loc[d, 'idx'] = tempjson['idx']
#             sfhcase.loc[d, 'ID'] = tempjson['id']
#             sfhcase.loc[d, 'SFH'] = tempjson['sfh']
#             sfhcase.loc[d, 'SFHDate'] = tempjson['sfhdate']
#             sfhcase.loc[d, 'SFHStaff'] = tempjson['sfhstaff']
#             sfhcase.loc[d, 'EDD'] = tempjson['edd']
#             sfhcase.loc[d, 'GA'] = tempga
#             sfhcase.loc[d, 'SFHStr'] = tempgastring
#             sfhcase.loc[d, 'GAdw'] = tempgadw
#             sfhcase.loc[d, 'Centile'] = tempsfhcentile
#
#         sfhcase = sfhcase.dropna()
#         sfhcase = sfhcase.sort_values('GA')
#         sfhcase = sfhcase.reset_index(drop=True)
#         #sfhcase = sfhcase.drop('index', axis=1)
#
#         centile_str1 = sfhcase.loc[sfhcase.shape[0] - 1, 'SFHStr']
#         centile_str2 = sfhcase.loc[sfhcase.shape[0] - 2, 'SFHStr']
#         centile1 = sfhcase.loc[sfhcase.shape[0] - 1, 'Centile']
#         centile2 = sfhcase.loc[sfhcase.shape[0] - 2, 'Centile']
#         id_return = sfhcase.loc[sfhcase.shape[0] - 1, 'ID']
#         id_return = int(id_return)
#
#         gadw1 = sfhcase.loc[0, 'GAdw']
#         gadw2 = sfhcase.loc[sfhcase.shape[0] - 1, 'GAdw']
#
#         sfd = 0
#         if centile1 < 10:
#             sfd = 1
#
#         lfd = 0
#         if centile1 > 90 and centile2 > 90:
#             lfd = 1
#
#         plot1return, plot2return, plot_uid = get_plot(centile_str1, sfhcase, sfhdata, gadw1, gadw2, sfd, lfd)
#
#         jsonreturn = {"lfd": lfd, "sfd": sfd, "centstring": centile_str1, "plotuid": plot_uid,
#                       "plotstring": plot1return, "plotpdfstring": plot2return, "id": id_return}
#         jsonreturn = json.dumps(jsonreturn)
#         jsonreturn = json.loads(jsonreturn)
#
#         return jsonreturn


##
## Actually setup the Api resource routing here
##


api.add_resource(TodoList, '/todos')
api.add_resource(Todo, '/todos/<todo_id>')
#api.add_resource(GetSFH, '/getsfhchart')
api.add_resource(GACentile, '/ga')
#api.add_resource(StoreSFH, '/sfh')
#api.add_resource(GetUID, '/uid')
#api.add_resource(RetrieveSFH, '/getsfh')
#api.add_resource(GetQR, '/qrcode')

if __name__ == '__main__':
    app.run()