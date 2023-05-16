#pragma once

#include "Population.h"
#include <SFML/Graphics.hpp>
#include <string>
#include <iostream>

class Drawer {
private :
    sf::RenderWindow w;
    sf::Font font;
public:
    bool paused;
    Drawer(int w, int h) :
        w(sf::VideoMode(w, h), "Fittest Genotype")
    {
        paused = false;
        if (!font.loadFromFile("Roboto-Black.ttf"))
        {
            std::cerr << "THE DLL WAS COMPILED WITH THE 'DRAWING' PREPROCESSOR DIRECTIVE BUT THE SPECIFIED TEXT FONT (.ttf) WAS NOT FOUND. "
                << "MAKE SURE IT IS ALONGSIDE THE EXECUTABLE, OR IN THE ACTIVE DIRECTORY." << std::endl;
        }
    };

    void draw(Network* n, int step) {

        constexpr float nodeRadius = 10.0f;
        constexpr float wheelRadius = 40.0f;
        constexpr float offset = 130.0f;

        static sf::CircleShape node(10.0f);
        static sf::CircleShape selfConnexion(nodeRadius);
        static sf::Vertex line[] =
        {
            sf::Vertex(sf::Vector2f(0.0f, 0.0f)),
            sf::Vertex(sf::Vector2f(0.0f, 0.0f))
        };
        static sf::Text text;

        static std::vector<float> Xs(
            MAX_COMPLEX_CHILDREN_PER_COMPLEX + MAX_MEMORY_CHILDREN_PER_COMPLEX 
        );
        static std::vector<float> Ys(
            MAX_COMPLEX_CHILDREN_PER_COMPLEX + MAX_MEMORY_CHILDREN_PER_COMPLEX
        );

        w.clear();
        sf::Event event;
        while (w.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                w.close();
            else if (event.type == sf::Event::KeyReleased && event.key.code == sf::Keyboard::Space) {
                paused = !paused;
            }
        }


        text.setFont(font);
        text.setCharacterSize(15);
        text.setFillColor(sf::Color::White);
        text.setString("Blue circle = complex node      White circle = memory node");
        text.setPosition(10.0f, w.getSize().y - 30.0f);
        w.draw(text);
        text.setString("Generation " + std::to_string(step));
        text.setPosition(w.getSize().x - 150.0f, 10.0f);
        w.draw(text);

        selfConnexion.setFillColor(sf::Color::Transparent);
        selfConnexion.setOutlineColor(sf::Color::White);
        selfConnexion.setOutlineThickness(1.0f);

        if (paused) {
            text.setString("PAUSED !  Press SPACE to resume.");
            text.setPosition(10.0f, 10.0f);
            w.draw(text);
        }


        float x0 = offset/2.0f, y0 = offset/1.5f;
        for (int i = (int)n->complexGenome.size(); i >= 0; i--) {
            ComplexNode_G* gNode = i == n->complexGenome.size() ? n->topNodeG.get() : n->complexGenome[i].get();
            if (gNode->phenotypicMultiplicity == 0) {continue;}
            if (x0 + offset > w.getSize().x) {
                x0 = offset; 
                y0 = 2.25f * offset;
            }
            
            text.setFillColor(sf::Color::White);
            if (i == n->complexGenome.size()) {
                text.setString("Top node, depth " + std::to_string(gNode->depth));
            }
            else {
                text.setString("Node n°" + std::to_string(gNode->position)  + ", depth " + std::to_string(gNode->depth));
            }
            text.setPosition(x0-1.25f*wheelRadius, y0 + 1.75f * wheelRadius);
            w.draw(text);
            text.setString("In size " + std::to_string(gNode->inputSize) + ", out size " + std::to_string(gNode->outputSize));
            text.setPosition(x0- 1.25f * wheelRadius, y0 + 2.25f * wheelRadius);
            w.draw(text);

            int _nNodes = (int) (gNode->complexChildren.size() + gNode->memoryChildren.size()) ;
            float factor = 6.28f / (float) _nNodes;
            for (int j = 0; j < _nNodes; j++) {
                Xs[j] = x0 + wheelRadius * cosf(factor * (float)j);
                Ys[j] = y0 + wheelRadius * sinf(factor * (float)j);
            }


            
            text.setFillColor(sf::Color::Black);

            

            node.setFillColor(sf::Color::Cyan);
            for (int j = 0; j < gNode->complexChildren.size(); j++) {
       
                node.setPosition(Xs[j], Ys[j]);

                w.draw(node);

                text.setString(std::to_string(gNode->complexChildren[j]->position));
                text.setPosition(Xs[j] + nodeRadius / 4.0f, Ys[j] + nodeRadius / 4.0f);
                w.draw(text);
            }

            node.setFillColor(sf::Color::White);
            for (int j = 0; j < gNode->memoryChildren.size(); j++) {
                int id = (int) gNode->complexChildren.size() + j;
                node.setPosition(Xs[id], Ys[id]);

                w.draw(node);

                text.setString(std::to_string(gNode->memoryChildren[j]->position));
                text.setPosition(Xs[id] + nodeRadius / 4.0f, Ys[id] + nodeRadius / 4.0f);
                w.draw(text);
            }

            x0 += offset*1.25f;
        }

        w.display();
    }
};