head = '''
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border:none;border-color:#ccc;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;}
.tg .tg-0ord{text-align:right}
.tg .tg-s6z2{text-align:center}
.tg .tg-34fq{font-weight:bold;text-align:right}
</style>
<table class="tg">
  <tr>
    <th class="tg-0ord"></th>
    <th class="tg-s6z2" colspan="3">Precision</th>
    <th class="tg-s6z2" colspan="3">Recall<br></th>
    <th class="tg-s6z2" colspan="3">F1</th>
  </tr>
'''

bottom='''
</table>
'''

row = '''
  <tr>
    <td class="tg-34fq">%s</td>
    <td class="tg-s6z2">%.2f<br></td>
    <td class="tg-031e">%.2f</td>
    <td class="tg-031e">%.2f</td>
    <td class="tg-s6z2">%.2f</td>
    <td class="tg-031e">%.2f</td>
    <td class="tg-031e">%.2f</td>
    <td class="tg-s6z2">%.2f</td>
    <td class="tg-031e">%.2f</td>
    <td class="tg-031e">%.2f</td>
  </tr>
'''