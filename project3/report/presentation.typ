#set page(width: 16cm, height: 9cm, margin: (x: 1.0cm, y: 0.7cm))
#set text(font: "Liberation Sans", size: 12pt)
#set heading(numbering: none)
#set par(justify: false, leading: 0.75em)

#let slide(title, body) = {
  align(left + top, [
    #text(size: 20pt, weight: "bold")[#title]
    #v(0.22cm)
    #body
  ])
  pagebreak()
}

#slide(
  [Optimization Landscapes with Bio-Inspired Search],
  [
    #text(size: 16pt)[IT3708 Project 3]
    #v(0.15cm)
    #text(size: 12pt)[Thomas Hansen \<thomasq\@stud.ntnu.no\>]
    #v(0.28cm)

    #text(size: 12pt)[
      Goal: Compare SGA, NSGA-II, and ACO on feature-selection landscapes and a synthetic triangle benchmark.
    ]
  ],
)

#slide(
  [Problem And Research Questions],
  [
    #text(size: 12pt)[
      - How does landscape structure affect optimizer performance?
      - Which algorithm is most robust across datasets?

    ]
    #v(0.2cm)
    #text(size: 12pt)[
      We evaluate:
      - SGA (single-objective GA)
      - NSGA-II (multi-objective evolutionary)
      - ACO (swarm intelligence)
    ]
  ],
)

#slide(
  [Experimental Setup],
  [
    #text(size: 12pt)[
      Shared budget for fair comparison:
      - 10 seeds (1000-1009)
      - Population/ant count: 120
      - 150 generations/iterations
    ]
    #v(0.15cm)
    #text(size: 12pt)[
      Fitness (feature landscapes):
      - Accuracy reward
      - Penalty for selected features
      - Penalty for normalized training time
    ]
    #v(0.15cm)
    #text(size: 12pt)[
      Local optima detected using strict Hamming-1 neighborhood.
    ]
  ],
)

#slide(
  [Datasets And Landscapes],
  [
    #text(size: 12pt)[
      Feature-selection HDF5 landscapes:
      - 01_breast-w
      - 05_letter-r
      - 08_credit-a
      - 06_zoo
      - 10_hepatitis
    ]
    #v(0.15cm)
    #text(size: 12pt)[
      Synthetic benchmark:
      - Triangle function ($n = 16, m = 1, s = 4$)
    ]
  ],
)

#slide(
  [Key Quantitative Results (New Datasets)],
  [
    #set text(size: 10pt)
    #table(
      columns: (1.8fr, 1.2fr, 0.9fr, 0.9fr, 0.9fr),
      inset: 3pt,
      stroke: 0.4pt,
      align: center,
      table.header[
        Dataset
      ][
        Optimizer
      ][
        Best
      ][
        Mean
      ][
        Std
      ],
      [06-zoo], [SGA], [0.8663], [0.8643], [0.0042],
      [06-zoo], [NSGA-II], [0.9792], [0.9773], [0.0016],
      [06-zoo], [ACO], [0.9792], [0.9782], [0.0015],
      [10-hepatitis], [SGA], [0.8482], [0.8473], [0.0012],
      [10-hepatitis], [NSGA-II], [0.9375], [0.9289], [0.0063],
      [10-hepatitis], [ACO], [0.9323], [0.9298], [0.0030],
    )
    #v(0.12cm)
    #text(size: 11pt)[
      Result:
      - NSGA-II/ACO clearly outperform SGA on all landscapes.
      - In the Zoo dataset, both ACO and NSGA-II reach global optimum.
      - In the hepatitis dataset, only NSGA-II found the global optimum.
    ]
  ],
)

#slide(
  [Zoo Landscape Example],
  [
    #grid(
      columns: (1fr, 1fr),
      gutter: 0.75cm,
      [
        #image("../artifacts/batch/06-zoo_lr_F-aco/fitness_landscape_3d.png", width: 100%)
        #text(size: 10pt)[ACO 3D surface]
      ],
      [
        #image("../artifacts/batch/06-zoo_lr_F-nsga-ii/convergence_curve.png", width: 100%)
        #text(size: 10pt)[NSGA-II convergence]
      ],
    )
  ],
)

#slide(
  [Hepatitis Landscape Example],
  [
    #grid(
      columns: (1fr, 1fr),
      gutter: 0.75cm,
      [
        #image("../artifacts/batch/10-hepatitis_lr_F-aco/fitness_landscape_3d.png", width: 70%)
        #text(size: 10pt)[ACO 3D surface]
      ],
      [
        #image("../artifacts/batch/10-hepatitis_lr_F-nsga-ii/convergence_curve.png", width: 70%)
        #text(size: 10pt)[NSGA-II convergence]
      ],
    )
  ],
)

#slide(
  [Large asymmetric triangle landscape],
  [
    #text(size: 12pt)[
      - The triangle function has a single global optimum at 31 active bits (fn=6).
      - There are also local optima at 15 and 25 bits (fn=5).
      - Since the algorithms rewards fewer bits, search pressure is towards the opposite end of the landscape, thus those solutions (with more active bits) are never found.
    ]
  ],
)
#slide(
  [Conclusions],
  [
    #text(size: 12pt)[
      - No single optimizer is universally best, but NSGA-II and ACO dominated SGA in these feature landscapes.
      - Landscape structure strongly influenced convergence quality.
    ]
  ],
)
