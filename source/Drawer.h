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

    // TODO when there are to much nodes...
    void draw(Network* n, int step) {

        constexpr float nodeRadius = 12.0f;
        constexpr float nodeOffset = 6.0f * nodeRadius;
        constexpr float lineOffset = 7.0f * nodeRadius;

        static sf::CircleShape node(nodeRadius);
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
        text.setString("In blue, complex nodes. In red memory. in/out are input/output sizes. kD = kernel dimension, d = depth, pM = phenotypic multiplicity"); // Legend
        text.setPosition(10.0f, w.getSize().y - 30.0f);
        w.draw(text);
        text.setString("Generation " + std::to_string(step));
        text.setPosition(w.getSize().x - 150.0f, 10.0f);
        w.draw(text);


        if (paused) {
            text.setString("PAUSED !  Press SPACE to resume.");
            text.setPosition(10.0f, 10.0f);
            w.draw(text);
        }

        text.setCharacterSize(11);
        int nComplexNodes = (int)n->complexGenome.size() + 1;
        int nMemoryNodes = (int)n->memoryGenome.size();
        float memoryOffset = nodeOffset * ((float)nComplexNodes - 1.0f) / ((float)nMemoryNodes - 1.0f +.000001f);
        float x0 = 2.0f*nodeRadius, y0 = 3.0f*nodeRadius + 2 * lineOffset;

        // Connexions, then complexNodes and their legend, then memory nodes and their legends.

        for (int i = (int)n->complexGenome.size(); i >= 0; i--) {
            ComplexNode_G* cNode = i == n->complexGenome.size() ? n->topNodeG.get() : n->complexGenome[i].get();
            float x = x0 + (nComplexNodes - 1 - cNode->position) * nodeOffset;
            line[0].position.x = x + nodeRadius;
            line[0].position.y = y0 + nodeRadius;

            line[1].position.y = y0 - lineOffset + nodeRadius;
            for (int j = 0; j < cNode->complexChildren.size(); j++) {
                line[1].position.x = x0 + (nComplexNodes - 1 - cNode->complexChildren[j]->position) * nodeOffset + nodeRadius;
                w.draw(line, 2, sf::Lines);
            }

            line[1].position.y = y0 + lineOffset + nodeRadius;
            for (int j = 0; j < cNode->memoryChildren.size(); j++) {
                line[1].position.x = x0 + (nMemoryNodes - 1 - cNode->memoryChildren[j]->position) * memoryOffset + nodeRadius;
                w.draw(line, 2, sf::Lines);
            }
        }

        node.setFillColor(sf::Color::Cyan);
        for (int i = 0; i < nComplexNodes; i++) {
            ComplexNode_G* cNode = i == n->complexGenome.size() ? n->topNodeG.get() : n->complexGenome[i].get();

            float x = x0 + (nComplexNodes - 1 - cNode->position) * nodeOffset;

            node.setPosition(x, y0);
            w.draw(node);

            node.setPosition(x, y0 - lineOffset);
            w.draw(node);

            text.setString("in: " + std::to_string(cNode->inputSize) + " out: " + std::to_string(cNode->outputSize));
            text.setPosition(x - .5f * nodeRadius, y0 - lineOffset - 1.1f * nodeRadius);
            w.draw(text);
            text.setString("d: " + std::to_string(cNode->depth) + " pM: " + std::to_string(cNode->phenotypicMultiplicity));
            text.setPosition(x - .5f * nodeRadius, y0 - lineOffset - 2.6f * nodeRadius);
            w.draw(text);
        }

        node.setFillColor(sf::Color::Red);
        for (int i = 0; i < nMemoryNodes; i++) {
            MemoryNode_G* mn = n->memoryGenome[i].get();

            float x = x0 + (nMemoryNodes - 1 - mn->position) * memoryOffset;
            float y = y0 + lineOffset;

            node.setPosition(x, y);
            w.draw(node);

            text.setString("in: " + std::to_string(mn->inputSize) + " out: " + std::to_string(mn->outputSize));
            text.setPosition(x - .5f * nodeRadius,y + 2.1f * nodeRadius);
            w.draw(text);
            text.setString("kD: " + std::to_string(mn->kernelDimension) + " pm " + std::to_string(mn->phenotypicMultiplicity));
            text.setPosition(x - .5f * nodeRadius, y + 3.6f * nodeRadius);
            w.draw(text);
        }

        w.display();
    }
};