from pathlib import Path
import zipfile, re
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image
from pypdf import PdfReader
w=Path('btp_revision_work');out=Path('reports');name='Machine_Learning_in_Forex'
pages=w.joinpath('content.txt').read_text(encoding='utf-8-sig').strip().split('\n~\n')
for n,f in [('body','cambria.ttc'),('bold','cambriab.ttf'),('sans','segoeui.ttf')]:pdfmetrics.registerFont(TTFont(n,str(Path('C:/Windows/Fonts')/f)))
pdf=w/(name+'.pdf');c=Canvas(str(pdf),pagesize=A4);c.setTitle('Machine Learning in Forex');c.setAuthor('Prince Kumar');W,H=A4;L=48;width=W-2*L;teal='#14675e';ink='#203846'
def st(font,size,lead,color=ink,align=0):return ParagraphStyle('s',fontName=font,fontSize=size,leading=lead,textColor=HexColor(color),alignment=align)
def para(txt,sty,gap=8,off=0):
 global y
 p=Paragraph(txt,sty);_,h=p.wrap(width-off,900);p.drawOn(c,L+off,y-h);y-=h+gap
def tx(x):return x.replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('−','-').replace('×',r'$\times$').replace('·',r'$\cdot$').replace('–','--').replace('’',"'")
tex=[r'\documentclass[10pt,a4paper]{article}',r'\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage[margin=20mm]{geometry}\usepackage{graphicx,tabularx,booktabs,array,fancyhdr,enumitem}',r'\pagestyle{fancy}\fancyhf{}\lhead{IIT Indore}\rhead{Machine Learning in Forex}\cfoot{\thepage}\setlength{\headheight}{15pt}\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\setlist[itemize]{nosep,leftmargin=*}\begin{document}']
assets={}
for n,page in enumerate(pages,1):
 lines=page.splitlines();title=lines.pop(0);cv=n==1
 if not cv:tex.append(r'\clearpage')
 else:tex.append(r'\thispagestyle{empty}\begin{center}\vspace*{14mm}')
 tex.append(r'\section*{'+tx(title)+'}');c.bookmarkPage(str(n));c.addOutlineEntry(title,str(n));y=H-(115 if cv else 65)
 if not cv:
  c.setFillColor(HexColor(teal));c.setFont('sans',8);c.drawString(L,H-34,'INDIAN INSTITUTE OF TECHNOLOGY INDORE');c.drawRightString(W-L,H-34,'DEPARTMENT OF PHYSICS');c.setStrokeColor(HexColor('#adc5c0'));c.line(L,H-45,W-L,H-45)
 para(escape(title),st('bold',30 if cv else 21,37 if cv else 25,ink,int(cv)),17)
 while lines:
  v=lines.pop(0)
  if v.startswith('@'):
   p,cap,limit=v[1:].split('|');p=Path(p);a='figures/'+str(n)+'-'+p.name;assets[a]=p
   with Image.open(p) as im:ww,hh=im.size
   ratio=min(width/ww,float(limit)/hh);ww*=ratio;hh*=ratio;c.drawImage(str(p),(W-ww)/2,y-hh,ww,hh,mask='auto');y-=hh+6
   tex.append(r'{\centering\includegraphics[width=\linewidth,height='+str(round(float(limit)*.34))+r'mm,keepaspectratio]{'+a+r'}\par}')
   if cap:para(escape(cap),st('sans',9.1,12,teal),12);tex.append(r'{\small\itshape '+tx(cap)+r'\par}')
  elif '|' in v:
   rows=[v.split('|')]
   while lines and '|' in lines[0] and not lines[0].startswith('@'):rows.append(lines.pop(0).split('|'))
   k=len(rows[0]);a=.78 if n==2 else (.29 if n in [7,9,13] else (.47 if k==2 else .37));widths=[width*a]+[width*(1-a)/(k-1)]*(k-1)
   tbl=Table([[Paragraph(escape(s),st('sans' if j==0 else 'body',9.1 if j==0 else 10.4,13,'#FFFFFF' if j==0 else ink)) for s in row] for j,row in enumerate(rows)],colWidths=widths)
   tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor(teal)),('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor('#eaf2f1'),HexColor('#f7f9f8')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));_,h=tbl.wrap(width,900);tbl.drawOn(c,L,y-h);y-=h+13
   tex.append(r'\begin{tabularx}{\linewidth}{@{}p{'+str(round(a-.03,2))+r'\linewidth}'+('X'*(k-1))+r'@{}}\toprule')
   for j,row in enumerate(rows):tex.append(' & '.join(tx(s) for s in row)+r'\\'+(r'\midrule' if j==0 else ''))
   tex.append(r'\bottomrule\end{tabularx}\par')
  elif v.startswith('#'):
   para(escape(v[1:]),st('bold',14,18,teal,int(cv)),6);tex.append(r'\subsection*{'+tx(v[1:])+'}')
  elif v.startswith('- '):
   z=v[2:].split('. ',1);assert len(z)==2
   c.setFillColor(HexColor(teal));c.circle(L+3,y-6,2,fill=1,stroke=0)
   para('<font name="bold">'+escape(z[0])+'.</font> '+escape(z[1]),st('body',11.3,15.1),8,14)
   tex.append(r'\begin{itemize}\item \textbf{'+tx(z[0])+r'.} '+tx(z[1])+r'\end{itemize}')
  else:para(escape(v),st('body',14 if cv else 11.3,20 if cv else 15.1,ink,int(cv)),15 if cv else 10);tex.append(tx(v)+r'\par')
 if cv:tex.append(r'\end{center}')
 print(n,round(y))
 assert y>35,(n,y)
 if not cv:c.setFillColor(HexColor(ink));c.setFont('sans',8);c.drawString(L,26,'Machine Learning in Forex   /   Prince Kumar');c.drawRightString(W-L,26,str(n))
 c.showPage()
c.save();tex.append(r'\end{document}');zpath=w/(name+'_LaTeX.zip')
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
 z.writestr('main.tex','\n'.join(tex))
 for a,p in assets.items():z.write(p,a)
r=PdfReader(pdf);t=' '.join(p.extract_text() for p in r.pages);assert len(r.pages)==14 and len(assets)==12 and all(v in t for v in ('55.51','53.30','9,841.44','9,838.73','3.87'))
import shutil
for p in [pdf,zpath]:shutil.copy2(p,out/p.name)

