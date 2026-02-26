import bibtexparser
import os
import glob

path_lr = 'literature_review/texrefs'
path_ls = 'literature_synopsis/texrefs'

def gen_main_texs(tex, tex_paths):
  f = open(tex, 'r')
  lines = f.read().split('\n')
  
  with open(tex, 'w') as f:
    for l in lines:
      if(l != '% END'):
        f.write('%s\n' % l)
      else:
        f.write('\n'.join(tex_paths))
        f.write('% END\n')

with open('refs.bib') as bibtex_file:
  bib_database = bibtexparser.load(bibtex_file)

tex_lr_files = []
tex_ls_files = []
for k in bib_database.entries_dict.keys():
  author =  bib_database.entries_dict[k]['author'].split(' ')[0].split(',')[0]
  entry_title = bib_database.entries_dict[k]['title']
  entry_type = bib_database.entries_dict[k]['ENTRYTYPE']

  file_name = ('%s/%s.tex' % (path_lr, k)).replace('@', '_')
  if(not os.path.exists(file_name)):
    header_analysis = "%% --\n\section{\color{BrickRed}\\texttt{%s et. al.}, %s, \\texttt{%s}~\cite{%s}}\n" % (author, entry_title, k, k)
    with open(file_name, 'w') as f:
      f.write(header_analysis)
    cat_cmd = 'cat %s/emptyanalysis.tex.skl >> %s' % (path_lr, file_name)
    print(cat_cmd)
    os.system(cat_cmd)
    
    tex_lr_files.append('%% --\n\input{%s/%s}\n\n' % (path_lr, k.replace('@', '_')))

  content_synopsis = "  \\begin{itemize}\n    \item \\textit{Research Question:} \cmt{}\n    \item \\textit{Methodology:}\n    \item \\textit{Impressions:}\n  \end{itemize}\n"
  file_name = ('%s/%s.tex' % (path_ls, k)).replace('@', '_')
  if(not os.path.exists(file_name)):
    with open(file_name, 'w') as f:
      f.write('\section{\large \protect\\bibentry{%s} \cite{%s}}\n' % (k, k))
      f.write(content_synopsis)
  
    tex_ls_files.append('%% --\n\input{%s/%s}\n\n' % (path_ls, k.replace('@', '_')))

gen_main_texs('literature_review/literature_review.tex', tex_lr_files)
gen_main_texs('literature_synopsis/literature_synopsis.tex', tex_lr_files)
