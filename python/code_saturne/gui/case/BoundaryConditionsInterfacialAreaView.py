# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------------

# This file is part of code_saturne, a general-purpose CFD tool.
#
# Copyright (C) 1998-2023 EDF S.A.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
# Street, Fifth Floor, Boston, MA 02110-1301, USA.

#-------------------------------------------------------------------------------

"""
This module contains the following classes:
- BoundaryConditionsInterfacialAreaView
"""

#-------------------------------------------------------------------------------
# Standard modules
#-------------------------------------------------------------------------------

import string, logging

#-------------------------------------------------------------------------------
# Third-party modules
#-------------------------------------------------------------------------------

from code_saturne.gui.base.QtCore    import *
from code_saturne.gui.base.QtGui     import *
from code_saturne.gui.base.QtWidgets import *

# -------------------------------------------------------------------------------
# Application modules import
# -------------------------------------------------------------------------------

from code_saturne.gui.case.BoundaryConditionsInterfacialArea import Ui_BoundaryConditionsInterfacialArea

from code_saturne.model.Common import GuiParam
from code_saturne.gui.base.QtPage import DoubleValidator, from_qvariant
from code_saturne.model.InterfacialAreaModel import InterfacialAreaModel
from code_saturne.model.MainFieldsModel import MainFieldsModel
from code_saturne.model.InterfacialForcesModel import InterfacialForcesModel

# -------------------------------------------------------------------------------
# log config
# -------------------------------------------------------------------------------

logging.basicConfig()
log = logging.getLogger("BoundaryConditionsInterfacialAreaView")
log.setLevel(GuiParam.DEBUG)


# -------------------------------------------------------------------------------
# Main class
#-------------------------------------------------------------------------------

class BoundaryConditionsInterfacialAreaView(QWidget, Ui_BoundaryConditionsInterfacialArea) :
    """
    Boundary condition for interfacial area
    """
    def __init__(self, parent):
        """
        Constructor
        """
        QWidget.__init__(self, parent)

        Ui_BoundaryConditionsInterfacialArea.__init__(self)
        self.setupUi(self)

        # Connections
        self.lineEditDiameter.textChanged[str].connect(self.__slotDiameter)

        validatorDiam = DoubleValidator(self.lineEditDiameter, min = 0.)
        validatorDiam.setExclusiveMin(True)

        self.lineEditDiameter.setValidator(validatorDiam)


    def setup(self, case, fieldId):
        """
        Setup the widget
        """
        self.case = case
        self.__boundary = None
        self.__currentField = fieldId


    def showWidget(self, boundary):
        """
        Show the widget
        """
        self.__boundary = boundary

        interfacial_area_model = None
        dispersed_fields = MainFieldsModel(self.case).getDispersedFieldList() + InterfacialForcesModel(
            self.case).getGLIMfields()
        if dispersed_fields != []:
            interfacial_area_model = InterfacialAreaModel(self.case).getAreaModel(self.__currentField)

        if interfacial_area_model not in [None, "constant"]:
            self.lineEditDiameter.show()
            val = boundary.getDiameter(self.__currentField)
            self.lineEditDiameter.setText(str(val))
            self.show()
        else:
            self.hideWidget()


    def hideWidget(self):
        """
        Hide the widget
        """
        self.hide()


    @pyqtSlot(str)
    def __slotDiameter(self, text):
        """
        INPUT fraction value
        """
        if self.lineEditDiameter.validator().state == QValidator.Acceptable:
            value = from_qvariant(text, float)
            self.__boundary.setDiameter(self.__currentField, value)


#-------------------------------------------------------------------------------
# End
#-------------------------------------------------------------------------------
