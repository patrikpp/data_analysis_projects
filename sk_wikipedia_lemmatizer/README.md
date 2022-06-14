# Sk-Wiki Lemmatizer

Cieľom projektu je vytvoriť lemmatizer založený na anchor textoch zo slovenskej wikipédie. Príklad:

    vstup: mestom
    výstup: mesto

Dataset obsahuje pre každu wiki stranku element page, ktorý obsahuje elementy title, id a revision. V elemente revision sa nachadza element text, ktorý obsahuje text wiki stranky vo formáte Wikitext, a teda element text obsahuje anchor texty. Vo formáte Wikitext sa zvyčajne používa pre zápis anchor textov  syntax: [[link|anchor text]] napr. [[Slovenská národná rada (1918 – 1919)|Slovenskej národnej rady]]. Ďalší možný zápis pre anchor texty je [[link]] napr. [[električka]], a teda anchor text a link sa zhodujú alebo zápis [[trolejbus]]y. Anchor texty vyextrahujeme pomocou regulárnych výrazov. Anchor texty bude potrebne ešte vyčistiť, pretože obsahujú HTML znakové entity napr. [[súborový systém proc|súborový systém \&quot;/proc\&quot;]], kde \&quot; označuje znak ". Z extrahovaných anchor textov dekodujeme HTML znakové entity na obyčajný text.

Následne ešte prebehne tokenizovanie a lematizovanie týchto vyparsovaných dát. Po tokenizovaní získame jednotlivé slová z linku a anchor textu a z týchto slov sa ešte odstrania stop slová. Na lematizovanie týchto  tokenizovaných slov z anchor textu sa využívajú tokenizované slová z linku. Na základe mierne modifikovaného algoritmu pre výpočet Levenshteinovej  vzdialenosti, sa zisti podobnosť medzi odvodeným slovom z anchor textu a slovom v základnom tvare z linku (lemma). 

Link na dataset: https://dumps.wikimedia.org/skwiki/latest/skwiki-latest-pages-articles.xml.bz2.