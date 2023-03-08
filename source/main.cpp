#pragma once



#include <iostream>

#include <SFML/Graphics.hpp>
#include "sciplot/sciplot.hpp"

#include "Population.h"
#include "Random.h"

#define LOGV(v) for (const auto e : v) {cout << e << " ";}; cout << "\n"
#define LOG(x) cout << x << endl;

using namespace std;
using namespace sciplot;

// TODO  !
/*
Connecter les enfants � la neuromodulation, et pas le parent !
    
Moins important, essayer d'inverser l'ordre de propagation du signal de neuromodulation.
Plutot que de passer du parent aux enfants, passer de l'enfant aux connexions 
le concernant. Mais comment faire pour les simples neurones ?

//#define RISI_NAJARRO_2020
//#define USING_NEUROMODULATION
are the 2 mutually exclusive running modes. Change in Genotype.h.
*/

#define DRAWING


#ifdef DRAWING

class Drawer {
    sf::RenderWindow& w;
public:
    Drawer(sf::RenderWindow& w) : w(w) {};
    void draw(Network* n) {
        static sf::CircleShape node(10.0f);
        static sf::Vertex line[] =
        {
            sf::Vertex(sf::Vector2f(0.0f, 0.0f)),
            sf::Vertex(sf::Vector2f(0.0f, 0.0f))
        };

        constexpr float wheelRadius = 40.0f;
        constexpr float offset = 130.0f;
        
        static std::vector<float> Xs(MAX_CHILDREN_PER_BLOCK + 2);
        static std::vector<float> Ys(MAX_CHILDREN_PER_BLOCK + 2);


        float x0 = offset, y0 = offset;
        for (int i = n->genome.size() - 1; i >= n->nSimpleNeurons; i--) {
            
            node.setFillColor(sf::Color::Blue);

            float factor = 6.28 / ((float) n->genome[i]->children.size() + 2.0f);
            for (int j = 0; j < n->genome[i]->children.size() + 2; j++) {
                Xs[j] = x0 + wheelRadius * cosf(factor * (float)j);     
                Ys[j] = y0 + wheelRadius * sinf(factor * (float)j);
            }

            int oID, dID;
            for (int j = 0; j < n->genome[i]->childrenConnexions.size(); j++) {
                oID = n->genome[i]->childrenConnexions[j].originID;
                oID = oID == -1 ? n->genome[i]->children.size() + 1: oID;
                dID = n->genome[i]->childrenConnexions[j].destinationID;

                line[0].position.x = Xs[oID] + 10.0f; // + circle radius
                line[0].position.y = Ys[oID] + 10.0f;
                line[1].position.x = Xs[dID] + 10.0f;
                line[1].position.y = Ys[dID] + 10.0f;

                w.draw(line, 2, sf::Lines);
            }

            for (int j = 0; j < n->genome[i]->children.size(); j++) {
                node.setPosition(sf::Vector2f(Xs[j], Ys[j]));
                if (n->genome[i]->children[j]->isSimpleNeuron) {
                    node.setFillColor(sf::Color::White);
                } 
                else {
                    node.setFillColor(sf::Color::Blue);
                }
                w.draw(node);
            }

            node.setFillColor(sf::Color::Red); //output
            node.setPosition(sf::Vector2f(Xs[n->genome[i]->children.size()], Ys[n->genome[i]->children.size()]));
            w.draw(node);

            node.setFillColor(sf::Color::Green); //input
            node.setPosition(sf::Vector2f(Xs[n->genome[i]->children.size()+1], Ys[n->genome[i]->children.size()+1]));
            w.draw(node);
            
            x0 += offset;
        }
    }
};

#endif 

int main()
{

#ifdef DRAWING
    sf::RenderWindow window(sf::VideoMode(720, 480), "Top Node");
    Drawer drawer(window);
#endif

    int nThreads = std::thread::hardware_concurrency();
    int N_SPECIMENS = nThreads * 64;
    int nDifferentTrials = 5;
    int nSteps = 5000;


    LOG(nThreads << " concurrent threads are supported at hardware level. Using " << nThreads << ".");

    // ALL TRIALS MUST HAVE SAME netInSize AND netOutSize
    vector<Trial*> trials;
    for (int i = 0; i < nDifferentTrials; i++) {
        trials.push_back(new CartPoleTrial());
        //trials.push_back(new XorTrial(2));
    }

    Population population(trials[0]->netInSize, trials[0]->netOutSize, N_SPECIMENS);

    LOG("N_SPECIMEN = " << N_SPECIMENS << " and N_TRIALS = " << nDifferentTrials);

   /* Vec x = linspace(1, N_SPECIMENS, N_SPECIMENS);
    Plot2D plot;

    plot.xlabel("x");
    plot.ylabel("y");
    plot.xrange(0.0, N_SPECIMENS);
    plot.yrange(-2.0, 2.0);

    plot.drawCurve(x, population.).label("sin(x)");

    Figure fig = { {plot} };
    Canvas canvas = { {fig} };
    canvas.show();*/

    population.startThreads(nThreads);
    for (int i = 0; i < nSteps; i++) {
#ifdef DRAWING
        window.clear();
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        drawer.draw(population.getFittestSpecimenPointer());
#endif

        population.step(trials);
        if (i % 100 == 0) { // defragmentate.
            string fileName = population.save();
            population.load(fileName);
        }
        
#ifdef DRAWING
        window.display();
#endif
    }
    population.stopThreads();
    
    Network* n = population.getFittestSpecimenPointer();
    trials[0]->reset();
    n->intertrialReset();
    cout << "\n";
    while (!trials[0]->isTrialOver) {
        n->step(trials[0]->observations);
        trials[0]->step(n->getOutput());
        cout << ", " << trials[0]->observations[0] << ", " << trials[0]->observations[2] ;
    }
    cout << "\n" << trials[0]->score;

    return 0;
}
