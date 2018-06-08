head = '''
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border:none;border-color:#ccc;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;}
.tg .tg-88nc{font-weight:bold;border-color:inherit;text-align:center}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-uys7{border-color:inherit;text-align:center}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-uys7"></th>
    <th class="tg-88nc" colspan="3">Precision</th>
    <th class="tg-88nc" colspan="3">Recall<br></th>
    <th class="tg-88nc" colspan="3">F1</th>
  </tr>
  <tr>
    <td class="tg-baqh"></td>
    <td class="tg-amwm">all</td>
    <td class="tg-amwm">single</td>
    <td class="tg-amwm">multi</td>
    <td class="tg-amwm">all</td>
    <td class="tg-amwm">single</td>
    <td class="tg-amwm">multi</td>
    <td class="tg-amwm">all</td>
    <td class="tg-amwm">single</td>
    <td class="tg-amwm">multi</td>
  </tr>
'''

bottom='''
</table>
'''

row = '''
  <tr>
    <td class="tg-88nc">%s</td>
    <td class="tg-uys7">%.2f<br></td>
    <td class="tg-uys7">%.2f</td>
    <td class="tg-uys7">%.2f</td>
    <td class="tg-uys7">%.2f</td>
    <td class="tg-uys7">%.2f</td>
    <td class="tg-uys7">%.2f</td>
    <td class="tg-uys7">%.2f</td>
    <td class="tg-uys7">%.2f</td>
    <td class="tg-uys7">%.2f</td>
  </tr>
'''