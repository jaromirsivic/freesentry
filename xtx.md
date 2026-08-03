Popis teď budu diktovat takže může být plný překlepů při přepisu mluveného slova do textu. Vytvoř zadání pro umělou inteligenci jako seniorní IT analytik (expert na python se specializací na pytorch a vše co k tomu patří). Zamysli se nad všemi důsledky projektu, který budu popisovat a domysli zadání do největšího detailu, tak aby je AI dokázala bezchybně naimplementovat. Zadání:
Vytvoř nový projekt pro natrénování neuronové sítě a následnou inference neuronové sítě pro rozpoznání pózy člověka. Inspiruj se modelem "YOLO26n-pose", "YOLO26m-pose" a "YOLO26l-pose". Architekturu neuronové sítě těchto modelů nastuduj ze všech dostupných zdrojů, které máš, nebo které nalezneš na internetu. Nastuduj architekturu sítě důsledně a pedantsky, protože budeme vytvářet tři varianty modelu které budou mít podobnou, nebo lepší výpočetní složitost a budou přesnější, ale neuronová síť bude trošičku jiná. Náš model nazvaný XTX2 taktéž ve verzi "n", "m" a "l" bude zaměřený na extrémně rychlou detekci pózy člověka. Cílem je provozovat model ve variantě "XTX2 n" na Raspberry Pi. Ostatní varianty modelu budou provozovány na klasickém počítači. Knihovna/balík který vyvíjíme dostane na vstupu obrázek typicky numpy array a na výstupu vrátí "pose array" pole s pózami lidí, které byly na obrázku detekovány (inspiruj se formátem výstupu, který vrací YOLO26l-pose. Myslím, že póza člověka je definována 17 body a u každého bodu je ve výsledku vrácen confiedence level). Zde je popis toho jak bude obrázek na vstupu transformován do výstupu. Vstupní obrázek bude nejprve transformován do tří čtvercových obrázků vždy o rozlišení 384x384 pixelů. To znamená například pokud bude vstupní obrázek v rozlišení 1920x1080 tak se v level_1 (obrázek A) zmenší do rozlišení 1526x1526 se zachováním aspect ratio a zbylé oblasti se doplní černou barvou (tedy jelikož je obrázek na šířku, tak se doplní horní a dolní část, kdyby byl na výšku, tak by se doplnila levá a pravá část) a takto zmenšený obrázek se pak ještě jednou zmenší do velikosti 384x384. čtvercový obrázek level_2 pak vykopíruje střed o velikosti 768x768 pixelů z obrázku level_1 a takováto vykopírovaná oblast se zmenší do rozlišení 384x384 (pochopitelně toto je třeba naimplementovat na jeden shot, aby to bylo co nejrychlejší jak pro trénink, tak pro inference, obecně všechny 3 levely je třeba naimplementovat co nejefektivněji). Level_3 pak vykopíruje středovou oblast o rozlišení 384x384 z obrázku v level_2.
Kdyby byl původní obrázek ve velikost 320x240 tak se pro level_1 vytvoří skoro celý černý obrázek o rozlišení 1526x1526 a původní obrázek o rozlišení 320x240 se umístí doprostřed. Pak se postupuje dál s tvorbou obrázku level_2 a level_3.

Jakmile jsou vytvořeny všechny 3 levely, tak jsou všechny tři obrázky (level_1, level_2, level_3) zpracovány neuronovou sítí paralelně a pro každý z těchto tří obrázků jsou detekovány pózy lidí. jakmile detekce skončí tak je nutné určit jestli náhodou pózy detekované na obrázku level_3 nepatří stejným lidem jako pózy detekované na obrázku level_2 případně na obrázku level_1. Pokud ano pak póza detekovaná na obrázku level_1 má prioritu před pózou detekovanou na obrázku level_2, případně na obrázku level_3, tedy pózy stejných lidí se ve výsledném poli nesmí opakovat budou převzaty primárně z level_1, sekundárně z level_2 a terciálně z level_3. Cílem je získat co nejpřesnější pozice póz lidí uprostřed obrázku (tedy uprostřed originálního obrázku detekovat lidi, kteří jsou velmi vzdáleni od kamery a nebyli detekování v obrázku level_1 ani v obrázku level_2).
Při návrhu architektury neuronové sítě se můžeš inspirovat projektem YOLO, nebo můžeš přijít s něčím vlastním a výrazně lepším.
V rámci projektu vytvoř tréninkovou pipeline (bat soubor, který spustím, zeptá se mě jaký model chci trénovat (n, m, nebo l), kde jsou vstupní data a kam má ukládat checkpointy, případně se mě hned po spuštění zeptá jestli nechci pokračovat s tréninkem od nějakého checkpointu) a následně také samostatný příklad (bat soubor, který jednoduše spustím) pro inference.
Takže ještě jednou v tréninkové pipeline se Python program v konzolové aplikaci nejprve zeptá jaký model chce uživatel trénovat. Na výběr bude mít z varianty "n", "m" a "l", následně se program zeptá na cestu k tréninkovému datasetu defaultní hodnota je relativní cesta ./dataset, která je vybrána pokud uživatel stiskne enter. Adresář dataset by měl mít dvě podsložky "train" a "test". V každé z nich jsou uloženy obrázky a json soubory s anotacemi obrázků. Program prohledá všechny JSON soubory a zahájí trénink, průběžně vypisuje výsledky, a v předdefinovaném čase který je v root aplikace v souboru config.json ukládá checkpoint natrénované neuronové sítě. Defaultní hodnota pro checkpoint je 5 minut. Defaultní hodnota pro počet epoch je 50. Příklad JSON souboru s anotací obrázku je uveden níže:

{
  "images": [
    {
      "id": 1,
      "file_name": "img0001.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "name": "person",
      "supercategory": "human",
      "bbox": [0.234375, 0.166667, 0.3125, 0.791667],
      "iscrowd": 0,
      "num_keypoints": 17,
      "keypoint_names": [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
      ],
      "keypoints": [
        0.281250, 0.208333, 2,  
        0.273438, 0.197917, 2,   
        0.289062, 0.197917, 2,   
        0.265625, 0.218750, 1,  
        0.296875, 0.218750, 2,  
        0.250000, 0.270833, 2,  
        0.312500, 0.270833, 2,  
        0.234375, 0.375000, 2,  
        0.328125, 0.375000, 2,  
        0.218750, 0.458333, 2,  
        0.343750, 0.458333, 2,  
        0.257812, 0.541667, 2,  
        0.304688, 0.541667, 2,  
        0.250000, 0.708333, 2,  
        0.312500, 0.708333, 2,  
        0.242188, 0.875000, 0,  
        0.320312, 0.875000, 0   
      ]
    }
  ]
}

Důležité upozornění x, y hodnoty pózy v keypoints jsou normalizované, třetí hodnota v keypoints označuje viditelnost (0 - bod není vidět, 1 - bod je zakrytý, 2 - bod je jasně viditelný).
Zadání pro AI piš v angličtině a výsledek ulož do souboru persondetection2.md
Pokud se při vytváření popisu potřebuješ na cokoliv zeptat, tak se mě zeptej.

Dříve jsem již vytvářel neuronovou síť XTX (ketrá je uložena v adresáři XTX), ale chci ji výrazně vylepšit (zlepšit detekci a urychlit inference). Proto zde píši zadání pro XTX2. Implementaci původní XTX si můžeš prostudovat, ale moc se jí neřiď. Udělej to lépe. Zdrojové kódy ulož do adresáře xtx2.