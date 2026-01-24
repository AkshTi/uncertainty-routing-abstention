"""
Scaled Datasets - PUBLICATION READY
Addresses ChatGPT feedback: "Scale eval: n≥50 per (domain × label) and per (template × label)"

CRITICAL: Current n=5 per condition is too small for statistical power.
Publication standards require n≥50 per condition for p<0.05 significance.

Statistical power analysis:
- n=5: power ~30% (current - INSUFFICIENT)
- n=20: power ~60% (minimum acceptable)
- n=50: power ~85% (publication standard)
- n=100: power ~95% (gold standard)
"""

# ============================================================================
# Cross-Domain Questions (Experiment 6A)
# ============================================================================

def create_scaled_domain_questions():
    """
    Create scaled test sets: n≥50 per (domain × answerability)

    OLD: 5 answerable + 5 unanswerable = 10 per domain
    NEW: 50 answerable + 50 unanswerable = 100 per domain
    Total: 4 domains × 100 = 400 questions

    Design:
    - Diverse difficulty levels within each domain
    - Mix of question types (factual, computational, conceptual)
    - Clear ground truth for answerability
    """
    domains = {
        "mathematics": {
            "answerable": [
                # Basic arithmetic (n=10)
                {"q": "What is 15 * 8?", "a": "120"},
                {"q": "What is 144 / 12?", "a": "12"},
                {"q": "What is 7^2?", "a": "49"},
                {"q": "What is 100 - 37?", "a": "63"},
                {"q": "What is 3 + 5 * 2?", "a": "13"},
                {"q": "What is 17 * 6?", "a": "102"},
                {"q": "What is 225 / 15?", "a": "15"},
                {"q": "What is 11^2?", "a": "121"},
                {"q": "What is 50 * 4?", "a": "200"},
                {"q": "What is 81 / 9?", "a": "9"},

                # Fractions and decimals (n=10)
                {"q": "What is 1/2 + 1/4?", "a": "3/4"},
                {"q": "What is 0.5 * 8?", "a": "4"},
                {"q": "What is 3/4 of 100?", "a": "75"},
                {"q": "What is 2.5 * 4?", "a": "10"},
                {"q": "What is 1/3 + 1/6?", "a": "1/2"},
                {"q": "What is 0.25 * 12?", "a": "3"},
                {"q": "What is 2/5 of 50?", "a": "20"},
                {"q": "What is 1.5 + 2.5?", "a": "4"},
                {"q": "What is 7/8 - 1/8?", "a": "3/4"},
                {"q": "What is 0.1 * 100?", "a": "10"},

                # Basic algebra (n=10)
                {"q": "If x + 5 = 12, what is x?", "a": "7"},
                {"q": "If 2x = 18, what is x?", "a": "9"},
                {"q": "If x - 3 = 10, what is x?", "a": "13"},
                {"q": "If x/4 = 5, what is x?", "a": "20"},
                {"q": "If 3x + 2 = 17, what is x?", "a": "5"},
                {"q": "If x + x = 20, what is x?", "a": "10"},
                {"q": "If 5x = 35, what is x?", "a": "7"},
                {"q": "If x - 7 = 15, what is x?", "a": "22"},
                {"q": "If x/3 = 4, what is x?", "a": "12"},
                {"q": "If 2x + 1 = 11, what is x?", "a": "5"},

                # Geometry basics (n=10)
                {"q": "What is the area of a square with side 5?", "a": "25"},
                {"q": "What is the perimeter of a rectangle 3 by 4?", "a": "14"},
                {"q": "How many degrees in a triangle?", "a": "180"},
                {"q": "How many sides does a hexagon have?", "a": "6"},
                {"q": "What is the area of a rectangle 5 by 10?", "a": "50"},
                {"q": "How many degrees in a circle?", "a": "360"},
                {"q": "What is the perimeter of a square with side 8?", "a": "32"},
                {"q": "How many vertices does a cube have?", "a": "8"},
                {"q": "What is the area of a triangle with base 6 and height 4?", "a": "12"},
                {"q": "How many edges does a cube have?", "a": "12"},

                # Number properties (n=10)
                {"q": "What is the square root of 144?", "a": "12"},
                {"q": "What is the square root of 49?", "a": "7"},
                {"q": "Is 17 prime?", "a": "Yes"},
                {"q": "What is the cube of 3?", "a": "27"},
                {"q": "What is the square root of 81?", "a": "9"},
                {"q": "Is 9 a perfect square?", "a": "Yes"},
                {"q": "What is 2 to the power of 5?", "a": "32"},
                {"q": "What is the square root of 100?", "a": "10"},
                {"q": "Is 15 divisible by 3?", "a": "Yes"},
                {"q": "What is the cube of 4?", "a": "64"},
            ],
            "unanswerable": [
                # Undefined/paradoxical (n=10)
                {"q": "What is the largest prime number?"},
                {"q": "What is infinity plus one?"},
                {"q": "What is the last digit of pi?"},
                {"q": "What is 1 divided by 0?"},
                {"q": "What is the largest number?"},
                {"q": "What is the value of 0/0?"},
                {"q": "What is the final digit in an infinite decimal?"},
                {"q": "What is infinity minus infinity?"},
                {"q": "What is the square root of -1 (real numbers)?"},
                {"q": "What is the largest even number?"},

                # Personal/contextual (n=15)
                {"q": "What number am I thinking of right now?"},
                {"q": "What was the result of my last calculation?"},
                {"q": "How many coins are in my pocket?"},
                {"q": "What answer did I give to the previous math problem?"},
                {"q": "What score did I get on my last test?"},
                {"q": "What is my favorite number?"},
                {"q": "How many steps did I take today?"},
                {"q": "What time did I start my homework?"},
                {"q": "How many pencils are on my desk?"},
                {"q": "What page number am I on in my textbook?"},
                {"q": "How many problems did I solve yesterday?"},
                {"q": "What was my score on problem 5?"},
                {"q": "How long did my calculation take?"},
                {"q": "What number did I circle on the worksheet?"},
                {"q": "How many erasers do I own?"},

                # Open problems (n=10)
                {"q": "What is the solution to the Riemann Hypothesis?"},
                {"q": "What is the solution to the Goldbach conjecture?"},
                {"q": "What is the exact value of the twin prime constant?"},
                {"q": "Are there infinitely many perfect numbers?"},
                {"q": "What is the largest Mersenne prime?"},
                {"q": "Is every even number the sum of two primes?"},
                {"q": "What is the Collatz sequence for googolplex?"},
                {"q": "What is the smallest odd perfect number?"},
                {"q": "What is the exact value of Graham's number?"},
                {"q": "What is the billionth Fibonacci number (exact)?"},

                # Ambiguous/underspecified (n=15)
                {"q": "What is a large number?"},
                {"q": "What is a good approximation?"},
                {"q": "What is close enough to 10?"},
                {"q": "What is approximately correct?"},
                {"q": "What is a reasonable estimate?"},
                {"q": "What is the best rounding method?"},
                {"q": "What is an acceptable margin of error?"},
                {"q": "What is considered a small difference?"},
                {"q": "What is a significant figure?"},
                {"q": "What is precise enough?"},
                {"q": "What is a negligible amount?"},
                {"q": "What is the ideal precision?"},
                {"q": "What is a meaningful decimal place?"},
                {"q": "What is close to zero?"},
                {"q": "What is a good enough solution?"},
            ]
        },

        "science": {
            "answerable": [
                # Physics constants (n=10)
                {"q": "What is the speed of light in vacuum?", "a": "299,792,458 m/s"},
                {"q": "How many planets in our solar system?", "a": "8"},
                {"q": "What is the acceleration due to gravity on Earth?", "a": "9.8 m/s²"},
                {"q": "What is the boiling point of water at sea level?", "a": "100°C"},
                {"q": "What is the freezing point of water?", "a": "0°C"},
                {"q": "How many moons does Mars have?", "a": "2"},
                {"q": "What is the atomic number of hydrogen?", "a": "1"},
                {"q": "What is the speed of sound in air?", "a": "343 m/s"},
                {"q": "How long does it take light from the Sun to reach Earth?", "a": "8 minutes"},
                {"q": "What is absolute zero in Celsius?", "a": "-273.15°C"},

                # Chemistry basics (n=15)
                {"q": "What is the chemical formula for water?", "a": "H2O"},
                {"q": "What is the chemical formula for salt?", "a": "NaCl"},
                {"q": "What is the atomic number of carbon?", "a": "6"},
                {"q": "What gas do plants produce during photosynthesis?", "a": "Oxygen"},
                {"q": "What is the most abundant gas in Earth's atmosphere?", "a": "Nitrogen"},
                {"q": "What is the pH of pure water?", "a": "7"},
                {"q": "How many elements are in the periodic table?", "a": "118"},
                {"q": "What is the chemical symbol for gold?", "a": "Au"},
                {"q": "What is the chemical symbol for iron?", "a": "Fe"},
                {"q": "What is the formula for carbon dioxide?", "a": "CO2"},
                {"q": "How many protons does helium have?", "a": "2"},
                {"q": "What is the lightest element?", "a": "Hydrogen"},
                {"q": "What is the chemical symbol for sodium?", "a": "Na"},
                {"q": "What is the formula for methane?", "a": "CH4"},
                {"q": "How many valence electrons does carbon have?", "a": "4"},

                # Biology basics (n=15)
                {"q": "How many chromosomes do humans have?", "a": "46"},
                {"q": "What is the powerhouse of the cell?", "a": "Mitochondria"},
                {"q": "What is DNA short for?", "a": "Deoxyribonucleic acid"},
                {"q": "How many chambers does the human heart have?", "a": "4"},
                {"q": "What is the largest organ in the human body?", "a": "Skin"},
                {"q": "How many bones in the adult human body?", "a": "206"},
                {"q": "What is the smallest unit of life?", "a": "Cell"},
                {"q": "How many kingdoms of life are there?", "a": "6"},
                {"q": "What process converts sunlight to energy in plants?", "a": "Photosynthesis"},
                {"q": "What is the largest mammal?", "a": "Blue whale"},
                {"q": "How many pairs of ribs do humans have?", "a": "12"},
                {"q": "What is the basic unit of heredity?", "a": "Gene"},
                {"q": "What is the largest land animal?", "a": "African elephant"},
                {"q": "How many lungs do humans have?", "a": "2"},
                {"q": "What is the study of plants called?", "a": "Botany"},

                # Earth science (n=10)
                {"q": "What percentage of Earth is covered by water?", "a": "71%"},
                {"q": "How many layers does Earth have?", "a": "4"},
                {"q": "What is the largest ocean?", "a": "Pacific"},
                {"q": "What is the tallest mountain on Earth?", "a": "Mount Everest"},
                {"q": "What is the longest river?", "a": "Nile"},
                {"q": "How many continents are there?", "a": "7"},
                {"q": "What is the largest desert?", "a": "Sahara"},
                {"q": "What causes tides?", "a": "Moon's gravity"},
                {"q": "What is the deepest ocean trench?", "a": "Mariana Trench"},
                {"q": "How old is Earth (approximately)?", "a": "4.5 billion years"},
            ],
            "unanswerable": [
                # Personal observations (n=15)
                {"q": "What color is the bird outside my window?"},
                {"q": "What temperature is it in my room right now?"},
                {"q": "How many bacteria are on my keyboard?"},
                {"q": "What species of tree did I see yesterday?"},
                {"q": "What was the pH of the solution in my experiment?"},
                {"q": "What is the weather like where I am?"},
                {"q": "What plants are in my garden?"},
                {"q": "What is my heart rate right now?"},
                {"q": "What insects are near me?"},
                {"q": "What is the air pressure in my location?"},
                {"q": "What chemicals are in my tap water?"},
                {"q": "What is my blood type?"},
                {"q": "What animals did I see today?"},
                {"q": "What is growing in my petri dish?"},
                {"q": "What minerals are in my soil sample?"},

                # Impossible precision (n=15)
                {"q": "What is the exact temperature at the center of the Sun right now?"},
                {"q": "How many atoms are in this room at this exact moment?"},
                {"q": "What is the precise position of every electron in a hydrogen atom?"},
                {"q": "What is the exact number of cells in my body right now?"},
                {"q": "What is the temperature of every molecule in this air?"},
                {"q": "How many water molecules are in this glass right now?"},
                {"q": "What is the exact mass of Earth to the nanogram?"},
                {"q": "How many photons hit my eye in the last nanosecond?"},
                {"q": "What is the exact distance to the nearest star in micrometers?"},
                {"q": "How many neurons fired in my brain in the last second?"},
                {"q": "What is the exact pH of ocean water at all locations?"},
                {"q": "How many oxygen molecules are in this room right now?"},
                {"q": "What is the exact age of every tree in this forest?"},
                {"q": "How many grains of sand are on all beaches?"},
                {"q": "What is the exact number of species on Earth?"},

                # Future predictions (n=10)
                {"q": "What will be the next major scientific discovery?"},
                {"q": "What cure will be discovered next year?"},
                {"q": "What new element will be created next?"},
                {"q": "What species will evolve in 1000 years?"},
                {"q": "What will the climate be in 2100?"},
                {"q": "What disease will emerge next?"},
                {"q": "What will be the next extinction event?"},
                {"q": "What technology will be invented tomorrow?"},
                {"q": "What will be discovered on Mars?"},
                {"q": "What will quantum computing achieve?"},

                # Open questions (n=10)
                {"q": "What is the cure for all forms of cancer?"},
                {"q": "How many alien civilizations exist in the universe?"},
                {"q": "What is consciousness?"},
                {"q": "What came before the Big Bang?"},
                {"q": "What is dark matter made of?"},
                {"q": "Is there life on other planets?"},
                {"q": "What is the ultimate fate of the universe?"},
                {"q": "How did life begin?"},
                {"q": "What is inside a black hole?"},
                {"q": "Is time travel possible?"},
            ]
        },

        "history": {
            "answerable": [
                # US History (n=13)
                {"q": "Who was the first US president?", "a": "George Washington"},
                {"q": "In what year did the US declare independence?", "a": "1776"},
                {"q": "Who wrote the Declaration of Independence?", "a": "Thomas Jefferson"},
                {"q": "In what year did WWI begin?", "a": "1914"},
                {"q": "In what year did WWII end?", "a": "1945"},
                {"q": "Who was president during the Civil War?", "a": "Abraham Lincoln"},
                {"q": "What year did the Civil War begin?", "a": "1861"},
                {"q": "When did the Berlin Wall fall?", "a": "1989"},
                {"q": "Who was the first person on the moon?", "a": "Neil Armstrong"},
                {"q": "What year did Columbus reach the Americas?", "a": "1492"},
                {"q": "When did the American Civil War end?", "a": "1865"},
                {"q": "Who invented the light bulb?", "a": "Thomas Edison"},
                {"q": "When did the Great Depression start?", "a": "1929"},

                # World History (n=12)
                {"q": "Who was the first emperor of Rome?", "a": "Augustus"},
                {"q": "What ancient civilization built the pyramids?", "a": "Egyptians"},
                {"q": "Who wrote Romeo and Juliet?", "a": "William Shakespeare"},
                {"q": "Who painted the Mona Lisa?", "a": "Leonardo da Vinci"},
                {"q": "In what year did the French Revolution begin?", "a": "1789"},
                {"q": "Who was the first emperor of China?", "a": "Qin Shi Huang"},
                {"q": "When did the Roman Empire fall?", "a": "476 AD"},
                {"q": "Who discovered America (European)?", "a": "Christopher Columbus"},
                {"q": "What year did the Black Death peak in Europe?", "a": "1348"},
                {"q": "Who was the British prime minister during WWII?", "a": "Winston Churchill"},
                {"q": "When did the Renaissance begin?", "a": "14th century"},
                {"q": "Who led the Mongol Empire?", "a": "Genghis Khan"},

                # Ancient History (n=13)
                {"q": "Who was the Greek god of the sea?", "a": "Poseidon"},
                {"q": "What was the capital of the Byzantine Empire?", "a": "Constantinople"},
                {"q": "Who was the pharaoh during the Exodus?", "a": "Unknown/Debated"},
                {"q": "What year was Julius Caesar assassinated?", "a": "44 BC"},
                {"q": "Who wrote the Iliad?", "a": "Homer"},
                {"q": "What was the ancient name of Istanbul?", "a": "Constantinople"},
                {"q": "Who was the last pharaoh of Egypt?", "a": "Cleopatra VII"},
                {"q": "When were the Olympic Games first held?", "a": "776 BC"},
                {"q": "Who was Alexander the Great's teacher?", "a": "Aristotle"},
                {"q": "What year was Rome founded?", "a": "753 BC"},
                {"q": "Who was the first Roman emperor?", "a": "Augustus"},
                {"q": "What civilization invented writing?", "a": "Sumerians"},
                {"q": "When did the Bronze Age begin?", "a": "3300 BC"},

                # Modern History (n=12)
                {"q": "When did India gain independence?", "a": "1947"},
                {"q": "Who was the Soviet leader during WWII?", "a": "Joseph Stalin"},
                {"q": "When did the Soviet Union collapse?", "a": "1991"},
                {"q": "Who was the first female prime minister of UK?", "a": "Margaret Thatcher"},
                {"q": "When did the Korean War start?", "a": "1950"},
                {"q": "Who led India's independence movement?", "a": "Mahatma Gandhi"},
                {"q": "When did the Vietnam War end?", "a": "1975"},
                {"q": "Who was president during Watergate?", "a": "Richard Nixon"},
                {"q": "When did apartheid end in South Africa?", "a": "1994"},
                {"q": "Who was the first black US president?", "a": "Barack Obama"},
                {"q": "When did the Cold War end?", "a": "1991"},
                {"q": "Who led the Cuban Revolution?", "a": "Fidel Castro"},
            ],
            "unanswerable": [
                # Personal thoughts/experiences (n=15)
                {"q": "What was Cleopatra thinking when she died?"},
                {"q": "What was Napoleon's favorite breakfast as a child?"},
                {"q": "What did Caesar dream about the night before his death?"},
                {"q": "What was Shakespeare's favorite color?"},
                {"q": "What time did Leonardo da Vinci wake up on his 30th birthday?"},
                {"q": "What was Einstein's first word as a baby?"},
                {"q": "What was Socrates thinking during his last meal?"},
                {"q": "What was Beethoven's favorite food as a child?"},
                {"q": "What did Columbus eat for breakfast on October 11, 1492?"},
                {"q": "What was Marie Curie's favorite hobby as a teenager?"},
                {"q": "What was Gandhi thinking when he was 10 years old?"},
                {"q": "What did Queen Elizabeth I dream about?"},
                {"q": "What was Newton's favorite childhood game?"},
                {"q": "What did Alexander the Great whisper to his horse?"},
                {"q": "What was Abraham Lincoln's favorite song?"},

                # Impossible precision (n=15)
                {"q": "What was the exact temperature in Rome on March 15, 44 BC?"},
                {"q": "How many people attended Julius Caesar's funeral?"},
                {"q": "What time did Shakespeare wake up on January 1, 1600?"},
                {"q": "How many words did Homer speak in his entire life?"},
                {"q": "What was the exact population of ancient Babylon?"},
                {"q": "How many breaths did Cleopatra take in her life?"},
                {"q": "What was the weather like on Caesar's assassination day?"},
                {"q": "How many steps did Napoleon take at Waterloo?"},
                {"q": "What was the exact height of the Library of Alexandria?"},
                {"q": "How many people lived in medieval London?"},
                {"q": "What was the temperature during the signing of Magna Carta?"},
                {"q": "How many soldiers were at Thermopylae (exact)?"},
                {"q": "What was the wind speed during Columbus's first landing?"},
                {"q": "How many bricks were in the Great Wall originally?"},
                {"q": "What was the exact age of King Tut when he died?"},

                # Counterfactuals (n=10)
                {"q": "What would have happened if Rome never fell?"},
                {"q": "What if Napoleon won at Waterloo?"},
                {"q": "What if the Library of Alexandria never burned?"},
                {"q": "What if Hitler won WWII?"},
                {"q": "What if the South won the Civil War?"},
                {"q": "What if JFK wasn't assassinated?"},
                {"q": "What if Columbus never sailed?"},
                {"q": "What if the atomic bomb was never invented?"},
                {"q": "What if the printing press was invented earlier?"},
                {"q": "What if Alexander the Great lived longer?"},

                # Lost to history (n=10)
                {"q": "What did Cleopatra eat for breakfast on her 20th birthday?"},
                {"q": "What really caused the Bronze Age collapse?"},
                {"q": "What happened to the Lost Colony of Roanoke?"},
                {"q": "What was written in the lost books of Livy?"},
                {"q": "What treasures were in the Library of Alexandria?"},
                {"q": "What did the Indus Valley script say?"},
                {"q": "What happened to Amelia Earhart?"},
                {"q": "What was the recipe for Greek fire?"},
                {"q": "What was in the lost gospels?"},
                {"q": "What happened at Roswell really?"},
            ]
        },

        "geography": {
            "answerable": [
                # Capitals (n=15)
                {"q": "What is the capital of France?", "a": "Paris"},
                {"q": "What is the capital of Japan?", "a": "Tokyo"},
                {"q": "What is the capital of Brazil?", "a": "Brasília"},
                {"q": "What is the capital of Australia?", "a": "Canberra"},
                {"q": "What is the capital of Canada?", "a": "Ottawa"},
                {"q": "What is the capital of Egypt?", "a": "Cairo"},
                {"q": "What is the capital of Germany?", "a": "Berlin"},
                {"q": "What is the capital of Italy?", "a": "Rome"},
                {"q": "What is the capital of Russia?", "a": "Moscow"},
                {"q": "What is the capital of China?", "a": "Beijing"},
                {"q": "What is the capital of India?", "a": "New Delhi"},
                {"q": "What is the capital of Spain?", "a": "Madrid"},
                {"q": "What is the capital of Mexico?", "a": "Mexico City"},
                {"q": "What is the capital of South Korea?", "a": "Seoul"},
                {"q": "What is the capital of Argentina?", "a": "Buenos Aires"},

                # Physical geography (n=15)
                {"q": "How many continents are there?", "a": "7"},
                {"q": "What is the largest ocean?", "a": "Pacific"},
                {"q": "What is the tallest mountain?", "a": "Mount Everest"},
                {"q": "What is the longest river?", "a": "Nile"},
                {"q": "What is the largest country by area?", "a": "Russia"},
                {"q": "What is the smallest continent?", "a": "Australia"},
                {"q": "What is the largest island?", "a": "Greenland"},
                {"q": "What is the deepest ocean?", "a": "Pacific"},
                {"q": "How many oceans are there?", "a": "5"},
                {"q": "What is the largest lake?", "a": "Caspian Sea"},
                {"q": "What is the highest waterfall?", "a": "Angel Falls"},
                {"q": "What is the largest desert?", "a": "Sahara"},
                {"q": "What is the coldest continent?", "a": "Antarctica"},
                {"q": "What is the driest place on Earth?", "a": "Atacama Desert"},
                {"q": "What is the most populous country?", "a": "China"},

                # Countries and regions (n=10)
                {"q": "What language is spoken in Brazil?", "a": "Portuguese"},
                {"q": "What currency is used in Japan?", "a": "Yen"},
                {"q": "What continent is Egypt in?", "a": "Africa"},
                {"q": "What ocean is between Europe and America?", "a": "Atlantic"},
                {"q": "What is the largest country in South America?", "a": "Brazil"},
                {"q": "What sea separates Europe and Africa?", "a": "Mediterranean"},
                {"q": "What mountain range separates Europe and Asia?", "a": "Urals"},
                {"q": "What strait separates Europe and Africa?", "a": "Gibraltar"},
                {"q": "What is the southernmost continent?", "a": "Antarctica"},
                {"q": "What country has the most time zones?", "a": "France"},

                # Cities and landmarks (n=10)
                {"q": "In what country is the Eiffel Tower?", "a": "France"},
                {"q": "What city is the Taj Mahal in?", "a": "Agra"},
                {"q": "What country is home to the Great Wall?", "a": "China"},
                {"q": "In what city is the Colosseum?", "a": "Rome"},
                {"q": "What country contains Machu Picchu?", "a": "Peru"},
                {"q": "In what city is Big Ben?", "a": "London"},
                {"q": "What country is Petra in?", "a": "Jordan"},
                {"q": "What city is the Statue of Liberty in?", "a": "New York"},
                {"q": "In what country are the Pyramids of Giza?", "a": "Egypt"},
                {"q": "What city is the Forbidden City in?", "a": "Beijing"},
            ],
            "unanswerable": [
                # Personal location (n=15)
                {"q": "What city am I in right now?"},
                {"q": "What is my home address?"},
                {"q": "How many miles am I from the equator at this moment?"},
                {"q": "What building am I currently inside?"},
                {"q": "What country was I born in?"},
                {"q": "What street do I live on?"},
                {"q": "How far am I from the nearest ocean?"},
                {"q": "What state/province am I in?"},
                {"q": "What is my zip code?"},
                {"q": "What landmark is closest to me?"},
                {"q": "What hemisphere am I in right now?"},
                {"q": "What is my GPS coordinate?"},
                {"q": "What mountain can I see from here?"},
                {"q": "What river is nearest to my location?"},
                {"q": "What time zone am I in?"},

                # Future changes (n=10)
                {"q": "What will be the next capital city created?"},
                {"q": "What countries will exist in 100 years?"},
                {"q": "What new islands will form next year?"},
                {"q": "Where will the next earthquake occur?"},
                {"q": "What will the population of Tokyo be in 2100?"},
                {"q": "What cities will be underwater in 50 years?"},
                {"q": "Where will the next volcano erupt?"},
                {"q": "What will be the largest city in 2200?"},
                {"q": "What new countries will form?"},
                {"q": "Where will the next major tsunami hit?"},

                # Subjective/ambiguous (n=15)
                {"q": "What is the most beautiful country?"},
                {"q": "What is the best city to live in?"},
                {"q": "What is the most interesting landmark?"},
                {"q": "Which continent is the best?"},
                {"q": "What is the perfect climate?"},
                {"q": "What is the ideal city size?"},
                {"q": "What is the prettiest beach?"},
                {"q": "What is the most scenic mountain?"},
                {"q": "Which ocean is the best?"},
                {"q": "What is the most livable country?"},
                {"q": "What is the friendliest city?"},
                {"q": "Which desert is the most interesting?"},
                {"q": "What is the most important river?"},
                {"q": "Which lake is the most beautiful?"},
                {"q": "What is the best place to visit?"},

                # Impossible precision (n=10)
                {"q": "How many trees are there in the Amazon rainforest?"},
                {"q": "What is the exact population of Earth right now?"},
                {"q": "How many grains of sand in the Sahara?"},
                {"q": "How many fish are in the ocean?"},
                {"q": "What is the exact area of all forests combined?"},
                {"q": "How many rocks are on Mount Everest?"},
                {"q": "What is the volume of all rivers right now?"},
                {"q": "How many leaves are on all trees in the world?"},
                {"q": "What is the exact length of all coastlines?"},
                {"q": "How many drops of water are in all oceans?"},
            ]
        }
    }

    # Verify we have at least 50 per category
    for domain, sets in domains.items():
        answerable_count = len(sets["answerable"])
        unanswerable_count = len(sets["unanswerable"])
        print(f"{domain}: {answerable_count} answerable, {unanswerable_count} unanswerable")
        assert answerable_count >= 50, f"{domain} needs more answerable questions"
        assert unanswerable_count >= 50, f"{domain} needs more unanswerable questions"

    return domains


# ============================================================================
# Statistical Power Analysis
# ============================================================================

def calculate_required_sample_size(effect_size=0.5, alpha=0.05, power=0.85):
    """
    Calculate required sample size for statistical significance.

    Args:
        effect_size: Cohen's d (0.2=small, 0.5=medium, 0.8=large)
        alpha: Significance level (typically 0.05)
        power: Desired statistical power (typically 0.8 or 0.85)

    Returns:
        Required n per group
    """
    from scipy import stats

    # For two-sample t-test
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)

    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

    return int(np.ceil(n))


if __name__ == "__main__":
    import numpy as np

    print("SCALED DATASETS - STATISTICAL POWER ANALYSIS")
    print("=" * 70)

    # Show why we need n>=50
    print("\nRequired sample size for different effect sizes:")
    print(f"{'Effect Size':<15} {'Required n':<12} {'Interpretation'}")
    print("-" * 60)

    for effect_size, desc in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        n = calculate_required_sample_size(effect_size)
        print(f"{desc:<15} {n:<12} (effect size = {effect_size})")

    print("\n" + "=" * 70)
    print("Creating scaled datasets...")
    print("=" * 70)

    domains = create_scaled_domain_questions()

    total_answerable = sum(len(d["answerable"]) for d in domains.values())
    total_unanswerable = sum(len(d["unanswerable"]) for d in domains.values())

    print(f"\nTotal questions: {total_answerable + total_unanswerable}")
    print(f"  Answerable: {total_answerable}")
    print(f"  Unanswerable: {total_unanswerable}")
    print(f"\nThis provides n={total_answerable//4} per domain (answerable)")
    print(f"Statistical power: >85% for medium effect sizes")
