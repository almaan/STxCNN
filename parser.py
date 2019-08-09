#!/usr/bin/env python3

import argparse as arp


def parser(date):

    prs = arp.ArgumentParser(description = '\n'.join(["STxCNN",
                                                          "------",
                                                          f"Date : {date}"]
                                                        ))

    prs.add_argument("-dp","--data_pth",
                     required = True,
                     type = str,
                     help = ' '.join(['Path to data',
                                      'data directory']
                                    )
                    )

    prs.add_argument("-b","--batch_size",
                     required = False,
                     default = 64,
                     type = int,
                     help = ' '.join([''],
                                    )
                    )

    prs.add_argument("-s","--samples",
                     required = False,
                     default = None,
                     type = int,
                     help = ' '.join(['Number of arrays',
                                      'to use in training',
                                     ]
                                    )
                    )

    prs.add_argument("-nw","--num_workers",
                     required = False,
                     default = 6,
                     type = int,
                     help = ' '.join(['',
                                     ])
                    )

    prs.add_argument("-e","--epochs",
                     required = False,
                     default = 50,
                     type = int,
                     help = ' '.join(['',
                                     ]
                                    )
                    )

    prs.add_argument("-o","--output_dir",
                     required = False,
                     default = None,
                     type = str,
                     help = ' '.join(["",
                                     ]
                                    )
                    )

    prs.add_argument("-tp","--training_patients",
                     required = False,
                     default = None,
                     type = str,
                     nargs = '+',
                     help = ' '.join(["",
                                     ]
                                    )
                    )

    prs.add_argument("-dv","--device",
                     required = False,
                     default = None,
                     type = str,
                     help = ' '.join(["",
                                     ]
                                    )
                    )
    prs.add_argument("-pt","--p_train",
                         required = False,
                         default = 0.8,
                         type = float,
                         help = ' '.join(["",
                                         ]
                                        )
                        )


    return prs
