#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import codecs
import csv
import os.path
import platform
import sys
import subprocess

import cv2
import numpy as np

from sys import stdout
from functools import partial

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from libs import oneCsvFile

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip

        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from libs.combobox import ComboBox
from libs.resources import *
from libs.constants import *
from libs.utils import *
from libs.settings import Settings
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from libs.stringBundle import StringBundle
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.labelDialog import LabelDialog
from libs.colorDialog import ColorDialog
from libs.labelFile import LabelFile, LabelFileError, LabelFileFormat, getShapesFromCsvFaceSet
from libs.toolBar import ToolBar
from libs.pascal_voc_io import PascalVocReader
from libs.pascal_voc_io import XML_EXT
from libs.yolo_io import YoloReader
from libs.yolo_io import TXT_EXT
from libs.create_ml_io import CreateMLReader
from libs.create_ml_io import JSON_EXT
from libs.csv_io import CsvReader
from libs.csv_io import CSV_EXT
from libs.ustr import ustr
from libs.hashableQListWidgetItem import HashableQListWidgetItem

from Boxes.BR import BoxRecommender

__appname__ = 'labelImg'


class WindowMixin(object):

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar


class MainWindow(QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self, defaultFilename=None, defaultPrefdefClassFile=None, defaultSaveDir=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        self.csvFilePath = 'localfaceset.csv'
        self.subject_dictionary = {}
        self.subject_name_to_id_dictionary = {}
        self.path_dictionary = {}
        self.path_to_id_dictionary = {}
        self.getembs_dictionary = {}

        # Recommender
        self.box_recommender = BoxRecommender()
        self.saved_label_exist = False
        self.embs = []
        self.boxes = []
        self.embsDict = {}

        # Load setting in the main thread
        self.settings = Settings()
        self.settings.load()
        settings = self.settings

        # Load string bundle for i18n
        self.stringBundle = StringBundle.getBundle()
        getStr = lambda strId: self.stringBundle.getString(strId)

        # Save as Pascal voc xml
        self.defaultSaveDir = defaultSaveDir
        self.labelFileFormat = settings.get(SETTING_LABEL_FILE_FORMAT, LabelFileFormat.PASCAL_VOC)
        self.preProcessPath = settings.get(SETTING_PREPROCESSING_PATH)

        # For loading all image under a directory
        self.mImgList = []
        self.dirname = None
        self.labelHist = []
        self.lastOpenDir = None

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False
        self._beginner = True
        self.screencastViewer = self.getAvailableScreencastViewer()
        self.screencast = "https://youtu.be/p0nR2YsCY_U"

        # Load predefined classes to the list
        self.loadPredefinedClasses(defaultPrefdefClassFile)

        # Main widgets and related state.
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)

        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.prevLabelText = ''

        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)

        self.similarityThreshText = QLineEdit()
        self.similarityThreshLabel = QLabel()
        self.similarityThreshLabel.setText("similarity threshold :")

        self.secondThreshText = QLineEdit()
        self.secondThreshLabel = QLabel()
        self.secondThreshLabel.setText("second threshold :")

        self.prepathText = QLineEdit()
        self.prepathLabel = QLabel()
        self.prepathLabel.setText("pre of data path :")
        self.prepathButton = QPushButton()
        self.prepathButton.setStyleSheet('QPushButton {background-color: #6FC1DA; color: green;}')
        self.prepathButton.setText('Get prepath')
        self.prepathButton.clicked.connect(self.getprepath)
        self.prepathText.setText(settings.get(SETTING_PREPATH))

        self.similarityLayout = QHBoxLayout()
        self.similarityLayout.addWidget(self.similarityThreshLabel)
        self.similarityLayout.addWidget(self.similarityThreshText)

        self.secondThreshLayout = QHBoxLayout()
        self.secondThreshLayout.addWidget(self.secondThreshLabel)
        self.secondThreshLayout.addWidget(self.secondThreshText)

        self.prepathLayout = QHBoxLayout()
        self.prepathLayout.addWidget(self.prepathLabel)
        self.prepathLayout.addWidget(self.prepathText)
        self.prepathLayout.addWidget(self.prepathButton)

        self.recomImageSizeText = QLineEdit()
        self.recomImageSizeLabel = QLabel()
        self.recomImageSizeLabel.setText("recommended image's size :")

        self.recomImagePadText = QLineEdit()
        self.recomImagePadLabel = QLabel()
        self.recomImagePadLabel.setText("recommended image's pad :")

        self.recomSizeLayout = QHBoxLayout()
        self.recomSizeLayout.addWidget(self.recomImageSizeLabel)
        self.recomSizeLayout.addWidget(self.recomImageSizeText)

        self.recomPadLayout = QHBoxLayout()
        self.recomPadLayout.addWidget(self.recomImagePadLabel)
        self.recomPadLayout.addWidget(self.recomImagePadText)

        self.recomLayout = QHBoxLayout()
        self.recomLayout.setContentsMargins(0, 0, 0, 0)

        self.w = QDialog()
        self.buttons = []
        self.labels = []
        self.clickedItem = None
        self.image_size = 200
        self.image_pad = 10
        self.newName = None
        self.recomImages = []

        self.recomLayout.addWidget(self.w)
        recomContainer = QWidget()
        recomContainer.setLayout(self.recomLayout)

        self.faceSetTextLine = QLineEdit()
        self.faceSetOkButton = QPushButton()
        self.faceSetOkButton.setStyleSheet('QPushButton {background-color: #6FC1DA; color: green;}')
        self.faceSetOkButton.setText('Choose csv face set')
        self.faceSetOkButton.clicked.connect(self.getFacesetPath)
        faceSetLayout = QVBoxLayout()
        faceSetLayout.addWidget(self.faceSetTextLine)
        faceSetLayout.addWidget(self.faceSetOkButton)
        faceSetTextContainer = QWidget()
        faceSetTextContainer.setLayout(faceSetLayout)

        self.preProcessTextLine = QLineEdit()
        self.preProcessOkButton = QPushButton()
        self.preProcessOkButton.setStyleSheet('QPushButton {background-color: #00F1F0; color: blue;}')
        self.preProcessOkButton.setText('from dir')
        self.preProcessOkButton.setToolTip("Get embedings from directory images")
        self.preProcessOkButton.clicked.connect(self.preProcessingWithPath)
        self.preProcessCsvButton = QPushButton()
        self.preProcessCsvButton.setStyleSheet('QPushButton {background-color: #00F1F0; color: blue;}')
        self.preProcessCsvButton.setText('from csv')
        self.preProcessCsvButton.setToolTip("Get embedings from csv images")
        self.preProcessCsvButton.clicked.connect(self.preProcessingWithCsv)
        self.preProcessTextLine.setText(self.preProcessPath)

        preProcessLayout = QHBoxLayout()
        preProcessLayout.addWidget(self.preProcessTextLine)
        # preProcessLayout.addWidget(self.preProcessOkButton)
        preProcessLayout.addWidget(self.preProcessCsvButton)
        preProcessTextContainer = QWidget()
        preProcessTextContainer.setLayout(preProcessLayout)

        self.maskCheckBox = QCheckBox('mask')
        self.maskCheckBox.setChecked(False)
        self.maskCheckBox.stateChanged.connect(self.maskChanged)
        maskContainer = QVBoxLayout()
        maskContainer.addWidget(self.maskCheckBox)
        featurecontainer = QWidget()
        featurecontainer.setLayout(maskContainer)
        # Create a widget for using default label
        self.useDefaultLabelCheckbox = QCheckBox(getStr('useDefaultLabel'))
        self.useDefaultLabelCheckbox.setChecked(False)
        self.defaultLabelTextLine = QLineEdit()
        useDefaultLabelQHBoxLayout = QHBoxLayout()
        useDefaultLabelQHBoxLayout.addWidget(self.useDefaultLabelCheckbox)
        useDefaultLabelQHBoxLayout.addWidget(self.defaultLabelTextLine)
        useDefaultLabelContainer = QWidget()
        useDefaultLabelContainer.setLayout(useDefaultLabelQHBoxLayout)

        # Create a widget for edit and diffc button
        self.diffcButton = QCheckBox(getStr('useDifficult'))
        self.diffcButton.setChecked(False)
        self.diffcButton.stateChanged.connect(self.btnstate)
        self.editButton = QToolButton()
        self.editButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Add some of widgets to listLayout
        listLayout.addWidget(self.editButton)
        listLayout.addWidget(self.diffcButton)
        listLayout.addWidget(useDefaultLabelContainer)
        listLayout.addWidget(preProcessTextContainer)
        # listLayout.addWidget(faceSetTextContainer)

        # Create and add combobox for showing unique labels in group
        self.comboBox = ComboBox(self)
        listLayout.addWidget(self.comboBox)

        # Create and add a widget for showing current label items
        self.labelList = QListWidget()
        labelListContainer = QWidget()
        labelListContainer.setLayout(listLayout)
        # recomListContainer.setLayout(recomLayout)
        self.labelList.itemActivated.connect(self.labelSelectionChanged)
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)
        # Connect to itemChanged to detect checkbox changes.
        self.labelList.itemChanged.connect(self.labelItemChanged)
        listLayout.addWidget(self.labelList)

        self.dock = QDockWidget(getStr('boxLabelText'), self)
        self.dock.setObjectName(getStr('labels'))
        self.dock.setWidget(labelListContainer)

        self.recomDock = QDockWidget("Recommends", self)
        self.recomDock.setObjectName("recomms")
        self.recomDock.setWidget(recomContainer)

        self.featuresDock = QDockWidget("Features", self)
        self.featuresDock.setWidget(featurecontainer)

        recomSizeContainer = QWidget()
        recomSizeContainer.setLayout(self.recomSizeLayout)
        similarityContainer = QWidget()
        similarityContainer.setLayout(self.similarityLayout)
        secondThreshContainer = QWidget()
        secondThreshContainer.setLayout(self.secondThreshLayout)
        prepathContainer = QWidget()
        prepathContainer.setLayout(self.prepathLayout)
        recomPadContainer = QWidget()
        recomPadContainer.setLayout(self.recomPadLayout)
        configs = QVBoxLayout()
        configs.addWidget(recomSizeContainer)
        configs.addWidget(similarityContainer)
        configs.addWidget(secondThreshContainer)
        # configs.addWidget(recomPadContainer)
        configs.addWidget(prepathContainer)
        confWidget = QWidget()
        confWidget.setLayout(configs)
        self.similarityThreshText.setText('0.2')
        self.secondThreshText.setText('1')
        self.recomImagePadText.setText('10')
        self.recomImageSizeText.setText('200')
        self.configDock = QDockWidget("Constants", self)
        self.configDock.setObjectName("configs")
        self.configDock.setWidget(confWidget)

        self.fileListWidget = QListWidget()
        self.fileListWidget.itemDoubleClicked.connect(self.fileitemDoubleClicked)
        filelistLayout = QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)
        filelistLayout.addWidget(self.fileListWidget)
        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.filedock = QDockWidget(getStr('fileList'), self)
        self.filedock.setObjectName(getStr('files'))
        self.filedock.setWidget(fileListContainer)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)

        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.setDrawingShapeToSquare(settings.get(SETTING_DRAW_SQUARE, False))

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.filedock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.configDock)
        self.filedock.setFeatures(QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.recomDock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.featuresDock)
        self.dockFeatures = QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
        self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

        # Actions
        action = partial(newAction, self)
        quit = action(getStr('quit'), self.close,
                      'Ctrl+Q', 'quit', getStr('quitApp'))

        getpreprocessPath = action(getStr('preProcessPath'), self.getPreProcessPath,
                                   'Ctrl+Shift+p', 'preProcess', getStr('preProcessPath'))

        open = action(getStr('openFile'), self.openFile,
                      'Ctrl+O', 'open', getStr('openFileDetail'))

        opendir = action(getStr('openDir'), self.openDirDialog,
                         'Ctrl+u', 'open', getStr('openDir'))

        copyPrevBounding = action(getStr('copyPrevBounding'), self.copyPreviousBoundingBoxes,
                                  'Ctrl+v', 'paste', getStr('copyPrevBounding'))

        changeSavedir = action(getStr('changeSaveDir'), self.changeSavedirDialog,
                               'Ctrl+r', 'open', getStr('changeSavedAnnotationDir'))

        openAnnotation = action(getStr('openAnnotation'), self.openAnnotationDialog,
                                'Ctrl+Shift+O', 'open', getStr('openAnnotationDetail'))

        openNextImg = action(getStr('nextImg'), self.openNextImg,
                             'd', 'next', getStr('nextImgDetail'))

        openPrevImg = action(getStr('prevImg'), self.openPrevImg,
                             'a', 'prev', getStr('prevImgDetail'))

        verify = action(getStr('verifyImg'), self.verifyImg,
                        'space', 'verify', getStr('verifyImgDetail'))

        save = action(getStr('save'), self.saveFile,
                      'Ctrl+S', 'save', getStr('saveDetail'), enabled=False)

        def getFormatMeta(format):
            """
            returns a tuple containing (title, icon_name) of the selected format
            """
            if format == LabelFileFormat.PASCAL_VOC:
                return ('&PascalVOC', 'format_voc')
            elif format == LabelFileFormat.YOLO:
                return ('&YOLO', 'format_yolo')
            elif format == LabelFileFormat.CREATE_ML:
                return ('&CreateML', 'format_createml')
            elif format == LabelFileFormat.CSV:
                return ('&CSV', 'format_csv')
            elif format == LabelFileFormat.ONECSV:
                return ('&ONECSV', 'format_onecsv')

        save_format = action(getFormatMeta(self.labelFileFormat)[0],
                             self.change_format, 'Ctrl+',
                             getFormatMeta(self.labelFileFormat)[1],
                             getStr('changeSaveFormat'), enabled=True)

        saveAs = action(getStr('saveAs'), self.saveFileAs,
                        'Ctrl+Shift+S', 'save-as', getStr('saveAsDetail'), enabled=False)

        close = action(getStr('closeCur'), self.closeFile, 'Ctrl+W', 'close', getStr('closeCurDetail'))

        deleteImg = action(getStr('deleteImg'), self.deleteImg, 'Ctrl+Shift+D', 'close', getStr('deleteImgDetail'))

        resetAll = action(getStr('resetAll'), self.resetAll, None, 'resetall', getStr('resetAllDetail'))

        color1 = action(getStr('boxLineColor'), self.chooseColor1,
                        'Ctrl+L', 'color_line', getStr('boxLineColorDetail'))

        createMode = action(getStr('crtBox'), self.setCreateMode,
                            'w', 'new', getStr('crtBoxDetail'), enabled=False)
        editMode = action('&Edit\nRectBox', self.setEditMode,
                          'Ctrl+J', 'edit', u'Move and edit Boxs', enabled=False)

        create = action(getStr('crtBox'), self.createShape,
                        'w', 'new', getStr('crtBoxDetail'), enabled=False)
        delete = action(getStr('delBox'), self.deleteSelectedShape,
                        'Delete', 'delete', getStr('delBoxDetail'), enabled=False)
        copy = action(getStr('dupBox'), self.copySelectedShape,
                      'Ctrl+D', 'copy', getStr('dupBoxDetail'),
                      enabled=False)

        advancedMode = action(getStr('advancedMode'), self.toggleAdvancedMode,
                              'Ctrl+Shift+A', 'expert', getStr('advancedModeDetail'),
                              checkable=True)

        hideAll = action('&Hide\nRectBox', partial(self.togglePolygons, False),
                         'Ctrl+H', 'hide', getStr('hideAllBoxDetail'),
                         enabled=False)
        showAll = action('&Show\nRectBox', partial(self.togglePolygons, True),
                         'Ctrl+A', 'hide', getStr('showAllBoxDetail'),
                         enabled=False)

        help = action(getStr('tutorial'), self.showTutorialDialog, None, 'help', getStr('tutorialDetail'))
        showInfo = action(getStr('info'), self.showInfoDialog, None, 'help', getStr('info'))

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action(getStr('zoomin'), partial(self.addZoom, 10),
                        'Ctrl++', 'zoom-in', getStr('zoominDetail'), enabled=False)
        zoomOut = action(getStr('zoomout'), partial(self.addZoom, -10),
                         'Ctrl+-', 'zoom-out', getStr('zoomoutDetail'), enabled=False)
        zoomOrg = action(getStr('originalsize'), partial(self.setZoom, 100),
                         'Ctrl+=', 'zoom', getStr('originalsizeDetail'), enabled=False)
        fitWindow = action(getStr('fitWin'), self.setFitWindow,
                           'Ctrl+F', 'fit-window', getStr('fitWinDetail'),
                           checkable=True, enabled=False)
        fitWidth = action(getStr('fitWidth'), self.setFitWidth,
                          'Ctrl+Shift+F', 'fit-width', getStr('fitWidthDetail'),
                          checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut,
                       zoomOrg, fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(getStr('editLabel'), self.editLabel,
                      'Ctrl+E', 'edit', getStr('editLabelDetail'),
                      enabled=False)
        self.editButton.setDefaultAction(edit)

        shapeLineColor = action(getStr('shapeLineColor'), self.chshapeLineColor,
                                icon='color_line', tip=getStr('shapeLineColorDetail'),
                                enabled=False)
        shapeFillColor = action(getStr('shapeFillColor'), self.chshapeFillColor,
                                icon='color', tip=getStr('shapeFillColorDetail'),
                                enabled=False)

        labels = self.dock.toggleViewAction()
        labels.setText(getStr('showHide'))
        labels.setShortcut('Ctrl+Shift+L')

        # Label list context menu.
        labelMenu = QMenu()
        addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu)

        # Draw squares/rectangles
        self.drawSquaresOption = QAction('Draw Squares', self)
        self.drawSquaresOption.setShortcut('Ctrl+Shift+R')
        self.drawSquaresOption.setCheckable(True)
        self.drawSquaresOption.setChecked(settings.get(SETTING_DRAW_SQUARE, False))
        self.drawSquaresOption.triggered.connect(self.toogleDrawSquare)

        # Store actions for further handling.
        self.actions = struct(save=save, save_format=save_format, saveAs=saveAs, open=open, close=close,
                              resetAll=resetAll, deleteImg=deleteImg,
                              lineColor=color1, create=create, delete=delete, edit=edit, copy=copy,
                              createMode=createMode, editMode=editMode, advancedMode=advancedMode,
                              shapeLineColor=shapeLineColor, shapeFillColor=shapeFillColor,
                              zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
                              fitWindow=fitWindow, fitWidth=fitWidth,
                              zoomActions=zoomActions,
                              fileMenuActions=(
                                  open, opendir, save, saveAs, close, resetAll, quit),
                              beginner=(), advanced=(),
                              editMenu=(edit, copy, delete,
                                        None, color1, self.drawSquaresOption),
                              beginnerContext=(create, edit, copy, delete),
                              advancedContext=(createMode, editMode, edit, copy,
                                               delete, shapeLineColor, shapeFillColor),
                              onLoadActive=(
                                  close, create, createMode, editMode),
                              onShapesPresent=(saveAs, hideAll, showAll))

        self.menus = struct(
            file=self.menu(getStr('menu_file')),
            edit=self.menu(getStr('menu_edit')),
            view=self.menu(getStr('menu_view')),
            help=self.menu(getStr('menu_help')),
            recentFiles=QMenu(getStr('menu_openRecent')),
            labelList=labelMenu)

        # Auto saving : Enable auto saving if pressing next
        self.autoSaving = QAction(getStr('autoSaveMode'), self)
        self.autoSaving.setCheckable(True)
        self.autoSaving.setChecked(settings.get(SETTING_AUTO_SAVE, False))
        # recommender : Enable auto detection
        self.recommenderMode = QAction(getStr('recommendMode'), self)
        self.recommenderMode.setCheckable(True)
        self.recommenderMode.setChecked(settings.get(SETTING_RECOMMENDER, False))
        # Sync single class mode from PR#106
        self.singleClassMode = QAction(getStr('singleClsMode'), self)
        self.singleClassMode.setShortcut("Ctrl+Shift+S")
        self.singleClassMode.setCheckable(True)
        self.singleClassMode.setChecked(settings.get(SETTING_SINGLE_CLASS, False))
        self.lastLabel = None
        # Add option to enable/disable labels being displayed at the top of bounding boxes
        self.displayLabelOption = QAction(getStr('displayLabel'), self)
        self.displayLabelOption.setShortcut("Ctrl+Shift+P")
        self.displayLabelOption.setCheckable(True)
        self.displayLabelOption.setChecked(settings.get(SETTING_PAINT_LABEL, False))
        self.displayLabelOption.triggered.connect(self.togglePaintLabelsOption)

        addActions(self.menus.file,
                   (open, opendir, getpreprocessPath, copyPrevBounding, changeSavedir, openAnnotation,
                    self.menus.recentFiles, save,
                    save_format, saveAs, close, resetAll, deleteImg, quit))
        addActions(self.menus.help, (help, showInfo))
        addActions(self.menus.view, (
            self.autoSaving,
            self.recommenderMode,
            self.singleClassMode,
            self.displayLabelOption,
            labels, advancedMode, None,
            hideAll, showAll, None,
            zoomIn, zoomOut, zoomOrg, None,
            fitWindow, fitWidth))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        addActions(self.canvas.menus[1], (
            action('&Copy here', self.copyShape),
            action('&Move here', self.moveShape)))

        self.tools = self.toolbar('Tools')
        self.actions.beginner = (
            open, opendir, changeSavedir, openNextImg, openPrevImg, verify, save, save_format, None, create, copy,
            delete, None,
            zoomIn, zoom, zoomOut, fitWindow, fitWidth)

        self.actions.advanced = (
            open, opendir, changeSavedir, openNextImg, openPrevImg, save, save_format, None,
            createMode, editMode, None,
            hideAll, showAll)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QImage()
        self.filePath = ustr(defaultFilename)
        self.lastOpenDir = None
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False
        # Add Chris
        self.difficult = False

        ## Fix the compatible issue for qt4 and qt5. Convert the QStringList to python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recentFileQStringList = settings.get(SETTING_RECENT_FILES)
                self.recentFiles = [ustr(i) for i in recentFileQStringList]
            else:
                self.recentFiles = recentFileQStringList = settings.get(SETTING_RECENT_FILES)

        size = settings.get(SETTING_WIN_SIZE, QSize(600, 500))
        position = QPoint(0, 0)
        saved_position = settings.get(SETTING_WIN_POSE, position)
        # Fix the multiple monitors issue
        for i in range(QApplication.desktop().screenCount()):
            if QApplication.desktop().availableGeometry(i).contains(saved_position):
                position = saved_position
                break
        self.resize(size)
        self.move(position)
        saveDir = ustr(settings.get(SETTING_SAVE_DIR, None))
        self.lastOpenDir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))
        if self.defaultSaveDir is None and saveDir is not None and os.path.exists(saveDir):
            self.defaultSaveDir = saveDir
            self.statusBar().showMessage('%s started. Annotation will be saved to %s' %
                                         (__appname__, self.defaultSaveDir))
            self.statusBar().show()

        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))
        Shape.line_color = self.lineColor = QColor(settings.get(SETTING_LINE_COLOR, DEFAULT_LINE_COLOR))
        Shape.fill_color = self.fillColor = QColor(settings.get(SETTING_FILL_COLOR, DEFAULT_FILL_COLOR))
        self.canvas.setDrawingColor(self.lineColor)
        # Add chris
        Shape.difficult = self.difficult

        def xbool(x):
            if isinstance(x, QVariant):
                return x.toBool()
            return bool(x)

        if xbool(settings.get(SETTING_ADVANCE_MODE, False)):
            self.actions.advancedMode.setChecked(True)
            self.toggleAdvancedMode()

        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Since loading the file may take some time, make sure it runs in the background.
        if self.filePath and os.path.isdir(self.filePath):
            self.queueEvent(partial(self.importDirImages, self.filePath or ""))
        elif self.filePath:
            self.queueEvent(partial(self.loadFile, self.filePath or ""))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # Display cursor coordinates at the right of status bar
        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)

        # Open Dir if deafult file
        if self.filePath and os.path.isdir(self.filePath):
            self.openDirDialog(dirpath=self.filePath, silent=True)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.canvas.setDrawingShapeToSquare(False)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            # Draw rectangle if Ctrl is pressed
            self.canvas.setDrawingShapeToSquare(True)

    ## Support Functions ##
    def set_format(self, save_format):
        if save_format == FORMAT_PASCALVOC:
            self.actions.save_format.setText(FORMAT_PASCALVOC)
            self.actions.save_format.setIcon(newIcon("format_voc"))
            self.labelFileFormat = LabelFileFormat.PASCAL_VOC
            LabelFile.suffix = XML_EXT

        elif save_format == FORMAT_YOLO:
            self.actions.save_format.setText(FORMAT_YOLO)
            self.actions.save_format.setIcon(newIcon("format_yolo"))
            self.labelFileFormat = LabelFileFormat.YOLO
            LabelFile.suffix = TXT_EXT

        elif save_format == FORMAT_CREATEML:
            self.actions.save_format.setText(FORMAT_CREATEML)
            self.actions.save_format.setIcon(newIcon("format_createml"))
            self.labelFileFormat = LabelFileFormat.CREATE_ML
            LabelFile.suffix = JSON_EXT

        elif save_format == FORMAT_CSV:
            self.actions.save_format.setText(FORMAT_CSV)
            self.actions.save_format.setIcon(newIcon("format_csv"))
            self.labelFileFormat = LabelFileFormat.CSV
            LabelFile.suffix = CSV_EXT
        elif save_format == FORMAT_ONECSV:
            self.actions.save_format.setText(FORMAT_ONECSV)
            self.labelFileFormat = LabelFileFormat.ONECSV

    def change_format(self):
        if self.labelFileFormat == LabelFileFormat.PASCAL_VOC:
            self.set_format(FORMAT_YOLO)
        elif self.labelFileFormat == LabelFileFormat.YOLO:
            self.set_format(FORMAT_CREATEML)
        elif self.labelFileFormat == LabelFileFormat.CREATE_ML:
            self.set_format(FORMAT_CSV)
        elif self.labelFileFormat == LabelFileFormat.CSV:
            self.set_format(FORMAT_PASCALVOC)
        else:
            raise ValueError('Unknown label file format.')
        self.setDirty()

    def noShapes(self):
        return not self.itemsToShapes

    def toggleAdvancedMode(self, value=True):
        self._beginner = not value
        self.canvas.setEditing(True)
        self.populateModeActions()
        self.editButton.setVisible(not value)
        if value:
            self.actions.createMode.setEnabled(True)
            self.actions.editMode.setEnabled(False)
            self.dock.setFeatures(self.dock.features() | self.dockFeatures)
        else:
            self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

    def populateModeActions(self):
        if self.beginner():
            tool, menu = self.actions.beginner, self.actions.beginnerContext
        else:
            tool, menu = self.actions.advanced, self.actions.advancedContext
        self.tools.clear()
        addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (self.actions.create,) if self.beginner() \
            else (self.actions.createMode, self.actions.editMode)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def setBeginner(self):
        self.tools.clear()
        addActions(self.tools, self.actions.beginner)

    def setAdvanced(self):
        self.tools.clear()
        addActions(self.tools, self.actions.advanced)

    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.labelList.clear()
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()
        self.labelCoordinates.clear()
        self.comboBox.cb.clear()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    def getAvailableScreencastViewer(self):
        osName = platform.system()

        if osName == 'Windows':
            return ['C:\\Program Files\\Internet Explorer\\iexplore.exe']
        elif osName == 'Linux':
            return ['xdg-open']
        elif osName == 'Darwin':
            return ['open']

    ## Callbacks ##
    def showTutorialDialog(self):
        subprocess.Popen(self.screencastViewer + [self.screencast])

    def showInfoDialog(self):
        from libs.__init__ import __version__
        msg = u'Name:{0} \nApp Version:{1} \n{2} '.format(__appname__, __version__, sys.version_info)
        QMessageBox.information(self, u'Information', msg)

    def createShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.actions.create.setEnabled(False)

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    def setCreateMode(self):
        assert self.advanced()
        self.toggleDrawMode(False)

    def setEditMode(self):
        assert self.advanced()
        self.toggleDrawMode(True)
        self.labelSelectionChanged()

    def updateFileMenu(self):
        currFilePath = self.filePath

        def exists(filename):
            return os.path.exists(filename)

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def editLabel(self):
        self.labelSelectionChanged()

    # Tzutalin 20160906 : Add file list and dock to move faster
    def fileitemDoubleClicked(self, item=None):
        currIndex = self.mImgList.index(ustr(item.text()))
        if currIndex < len(self.mImgList):
            filename = self.mImgList[currIndex]
            if filename:
                self.loadFile(filename)

    def maskChanged(self):
        if self.canvas.selectedShape is not None:
            self.canvas.selectedShape.mask = 1 if self.maskCheckBox.isChecked() else 0

    # Add chris
    def btnstate(self, item=None):
        """ Function to handle difficult examples
        Update on each object """
        if not self.canvas.editing():
            return

        item = self.currentItem()
        if not item:  # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count() - 1)

        difficult = self.diffcButton.isChecked()

        try:
            shape = self.itemsToShapes[item]
        except:
            pass
        # Checked and Update
        try:
            if difficult != shape.difficult:
                shape.difficult = difficult
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    # React to canvas signals.
    def shapeSelectionChanged(self, selected=False):
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.canvas.selectedShape
            if shape:
                self.shapesToItems[shape].setSelected(True)
            else:
                self.labelList.clearSelection()
        self.actions.delete.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)

    def addLabel(self, shape):
        shape.paintLabel = self.displayLabelOption.isChecked()
        item = HashableQListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        item.setBackground(QColor(255, 255, 255, 180))
        self.itemsToShapes[item] = shape
        self.shapesToItems[shape] = item
        self.labelList.addItem(item)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)
        self.updateComboBox()

    def remLabel(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapesToItems[shape]
        self.labelList.takeItem(self.labelList.row(item))
        del self.shapesToItems[shape]
        del self.itemsToShapes[item]
        self.updateComboBox()

    def loadLabels(self, shapes):
        s = []
        for label, points, line_color, fill_color, difficult in shapes:
            shape = Shape(label=label)
            for x, y in points:

                # Ensure the labels are within the bounds of the image. If not, fix them.
                x, y, snapped = self.canvas.snapPointToCanvas(x, y)
                if snapped:
                    self.setDirty()

                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            shape.close()
            s.append(shape)

            if line_color:
                shape.line_color = QColor(0, 0, 0, 0)
            else:
                shape.line_color = QColor(255, 255, 255, 180)

            if fill_color:
                shape.fill_color = QColor(0, 0, 0, 0)
            else:
                shape.fill_color = QColor(255, 255, 255, 180)

            self.addLabel(shape)
        self.updateComboBox()
        self.canvas.loadShapes(s)

    def updateComboBox(self):
        # Get the unique labels and add them to the Combobox.
        itemsTextList = [str(self.labelList.item(i).text()) for i in range(self.labelList.count())]

        uniqueTextList = list(set(itemsTextList))
        # Add a null row for showing all the labels
        uniqueTextList.append("")
        uniqueTextList.sort()

        self.comboBox.update_items(uniqueTextList)

    def saveLabels(self, annotationFilePath):
        annotationFilePath = ustr(annotationFilePath)
        if self.labelFile is None:
            self.labelFile = LabelFile()
            self.labelFile.verified = self.canvas.verified

        def format_shape(s):
            return dict(label=s.label,
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        points=[(p.x(), p.y()) for p in s.points],
                        # add chris
                        difficult=s.difficult)

        shapes = [format_shape(shape) for shape in self.canvas.shapes]
        # Can add differrent annotation formats here
        try:
            if self.labelFileFormat == LabelFileFormat.PASCAL_VOC:
                if annotationFilePath[-4:].lower() != ".xml":
                    annotationFilePath += XML_EXT
                self.labelFile.savePascalVocFormat(annotationFilePath, shapes, self.filePath, self.imageData,
                                                   self.lineColor.getRgb(), self.fillColor.getRgb())
            elif self.labelFileFormat == LabelFileFormat.YOLO:
                if annotationFilePath[-4:].lower() != ".txt":
                    annotationFilePath += TXT_EXT
                self.labelFile.saveYoloFormat(annotationFilePath, shapes, self.filePath, self.imageData, self.labelHist,
                                              self.lineColor.getRgb(), self.fillColor.getRgb())
            elif self.labelFileFormat == LabelFileFormat.CREATE_ML:
                if annotationFilePath[-5:].lower() != ".json":
                    annotationFilePath += JSON_EXT
                self.labelFile.saveCreateMLFormat(annotationFilePath, shapes, self.filePath, self.imageData,
                                                  self.labelHist, self.lineColor.getRgb(), self.fillColor.getRgb())

            elif self.labelFileFormat == LabelFileFormat.CSV:
                if annotationFilePath[-4:].lower() != ".csv":
                    annotationFilePath += CSV_EXT
                # self.labelFile.saveCsvFormat(annotationFilePath, shapes, self.filePath, self.imageData, self.labelHist,
                #                              self.lineColor.getRgb(), self.fillColor.getRgb())
                start, end = self.labelFile.saveOneCsvFile(self.canvas.shapes, self.filePath, self.imageData,
                                                           self.path_dictionary, self.prepathText.text())
                self.getEmbedingsFromShapes(start, end)
                self.make_path_id_dictionary()

            elif self.labelFileFormat == LabelFileFormat.ONECSV:
                self.labelFile.saveOneCsvFile(shapes, self.filePath, self.imageData, self.path_dictionary)

            else:
                self.labelFile.save(annotationFilePath, shapes, self.filePath, self.imageData,
                                    self.lineColor.getRgb(), self.fillColor.getRgb())
            # print('Image:{0} -> Annotation:{1}'.format(self.filePath, annotationFilePath))
            return True
        except LabelFileError as e:
            self.errorMessage(u'Error saving label data', u'<b>%s</b>' % e)
            return False

    def copySelectedShape(self):
        self.addLabel(self.canvas.copySelectedShape())
        # fix copy and delete
        self.shapeSelectionChanged(True)

    def comboSelectionChanged(self, index):
        text = self.comboBox.cb.itemText(index)
        for i in range(self.labelList.count()):
            if text == "":
                self.labelList.item(i).setCheckState(2)
            elif text != self.labelList.item(i).text():
                self.labelList.item(i).setCheckState(0)
            else:
                self.labelList.item(i).setCheckState(2)

    def getEmbedingsFromShapes(self, start, end):
        if os.path.exists('localembs.npy'):
            self.embsDict.update(np.load('localembs.npy', allow_pickle=True).item())
        filePath = self.filePath
        if self.prepathText.text() in filePath:
            filePath = filePath.split(self.prepathText.text())[-1][1:]
        len_embs = len(list(self.embsDict.keys()))
        newid = len_embs
        if start > -1:
            diff = end - start + 1
            for key in range(end + self.len_faceset + 1, len_embs):
                self.embsDict[key - diff] = self.embsDict[key]

            newid = len_embs - diff
        image = Image.open(filePath)
        image = np.array(image)

        for shape in self.canvas.shapes:
            points = []
            points.append(round(shape.points[0].x()))
            points.append(round(shape.points[0].y()))
            points.append(round(shape.points[2].x()))
            points.append(round(shape.points[2].y()))
            crop = cropImage(image, points, thresh=10)
            try:
                boxes, emb = self.box_recommender.detector.detectWithLandMark(crop,
                                                                              self.box_recommender.detector.detector)
            except:
                try:
                    crop = cropImage(image, points, 20)
                    crop = cv2.resize(crop, (112, 112, 3))
                    emb = self.box_recommender.detector.detectWithLandMark(crop)
                except:
                    emb = None
            self.embsDict[newid] = [emb, points, shape.userid]
            newid += 1

        np.save('localembs.npy', self.embsDict)

    def labelSelectionChanged(self):
        self.w.close()
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        # text = self.labelDialog.popUp(item.text())
        if self.canvas.selectedShape is None:
            return
        image = Image.open(self.filePath)
        image = np.array(image)
        # points = convertPointsToXY(self.canvas.selectedShape.points)
        points = []
        points.append(round(self.canvas.selectedShape.points[0].x()))
        points.append(round(self.canvas.selectedShape.points[0].y()))
        points.append(round(self.canvas.selectedShape.points[2].x()))
        points.append(round(self.canvas.selectedShape.points[2].y()))
        crop = cropImage(image, points, 10)
        emb, boxes = None, None
        try:
            boxes, emb = self.box_recommender.detector.detectWithLandMark(crop,
                                                                          self.box_recommender.detector.detector)
        except:
            try:
                crop = np.array(cropImage(image, points, 20))
                crop = cv2.resize(crop, (112, 112, 3))
                emb = self.box_recommender.detector.getEmbedings(crop)
            except:
                emb = None
        indexes = []

        try:
            thresh = float(self.similarityThreshText.text())
            secondThresh = float(self.secondThreshText.text())
            if 100 > thresh > 1:
                thresh = thresh / 100.
            if 100 >= secondThresh > 1:
                secondThresh = secondThresh / 100

        except:
            thresh = 0.1
            secondThresh = 1

        if emb is not None and os.path.exists('localembs.npy'):
            self.embsDict.update(np.load('localembs.npy', allow_pickle=True).item())
            j = 0
            for key in list(self.embsDict.keys()):
                key_emb, key_box, key_name_id = self.embsDict[key]
                id = key
                p = self.path_dictionary[id]
                if not os.path.exists(p):
                    if len(self.prepathText.text()) > 2:
                        p = os.path.join(os.path.normpath(self.prepathText.text()), p)

                if not os.path.exists(p) or key_emb is None or key_name_id == -1: continue
                a = cosine_similarity(emb, key_emb)

                if a[0][0] >= thresh:
                    indexes.append((a[0][0], key_box, p, key_name_id))
                    if a[0][0] >= secondThresh:
                        j += 1
                if j == 3: break

        self.w = QDialog()
        self.buttons = []
        self.labels = []
        self.clickedItem = None
        try:
            self.image_size = int(self.recomImageSizeText.text())
        except:
            self.image_size = 200
        try:
            self.image_pad = int(self.recomImagePadText.text())
        except:
            self.image_pad = 10
        self.newName = None
        self.recomLayout.addWidget(self.w)
        self.showWindow(indexes)

        if self.clickedItem is None:
            text = item.text()
            if self.newName is not None:
                text = self.newName[0]
                self.update_subjects(text)
        else:
            text = self.clickedItem
        if text is not None:
            item.setText(text)
            item.setBackground(QColor(100, 0, 0, 100))
            self.setDirty()
            self.updateComboBox()
        item = self.currentItem()
        if item and self.canvas.editing():
            self._noSelectionSlot = True
            self.canvas.selectShape(self.itemsToShapes[item])
            shape = self.itemsToShapes[item]
            # Add Chris
            self.diffcButton.setChecked(shape.difficult)
        if self.canvas.selectedShape.mask:
            self.maskCheckBox.setChecked(True)
        else:
            self.maskCheckBox.setChecked(False)

    def labelItemChanged(self, item):
        shape = self.itemsToShapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            shape.line_color = QColor(255, 255, 255, 255)
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    # Callback functions:
    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        if not self.useDefaultLabelCheckbox.isChecked() or not self.defaultLabelTextLine.text():
            if len(self.labelHist) > 0:
                self.labelDialog = LabelDialog(
                    parent=self, listItem=self.labelHist)

            # Sync single class mode from PR#106
            if not self.recommenderMode.isChecked():
                if self.singleClassMode.isChecked() and self.lastLabel:
                    text = self.lastLabel
                else:
                    text = self.labelDialog.popUp(text=self.prevLabelText)
                    self.lastLabel = text
            elif self.canvas.shapes[-1].label is not None:
                text = self.canvas.shapes[-1].label
            else:
                text = 'unknown'
                self.lastLabel = text
        else:
            text = self.defaultLabelTextLine.text()

        # Add Chris
        self.diffcButton.setChecked(False)
        if text is not None:
            self.prevLabelText = text
            generate_color = QColor(0, 0, 0, 180)
            shape = self.canvas.setLastLabel(text, generate_color, generate_color)
            self.addLabel(shape)
            if self.beginner():  # Switch to edit mode.
                self.canvas.setEditing(True)
                self.actions.create.setEnabled(True)
            else:
                self.actions.editMode.setEnabled(True)
            self.setDirty()

            if text not in self.labelHist:
                self.labelHist.append(text)
        else:
            # self.canvas.undoLastLine()

            self.canvas.resetAllLines()

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    def zoomRequest(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scrollArea.width()
        h = self.scrollArea.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.itemsToShapes.items():
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def loadFile(self, filePath=None):

        """Load the specified file, or the last opened file if None."""
        self.preProcessPath = self.preProcessTextLine.text()
        self.make_path_id_dictionary()
        self.make_subject_dictionary()
        self.update_embeding()
        self.w.close()
        self.resetState()
        self.canvas.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get(SETTING_FILENAME)

        # Make sure that filePath is a regular python string, rather than QString
        filePath = ustr(filePath)

        # Fix bug: An  index error after select a directory when open a new file.
        unicodeFilePath = ustr(filePath)
        unicodeFilePath = os.path.abspath(unicodeFilePath)
        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item
        if unicodeFilePath and self.fileListWidget.count() > 0:
            if unicodeFilePath in self.mImgList:
                index = self.mImgList.index(unicodeFilePath)
                fileWidgetItem = self.fileListWidget.item(index)
                fileWidgetItem.setSelected(True)
            else:
                self.fileListWidget.clear()
                self.mImgList.clear()

        if unicodeFilePath and os.path.exists(unicodeFilePath):
            if LabelFile.isLabelFile(unicodeFilePath):
                try:
                    self.labelFile = LabelFile(unicodeFilePath)
                except LabelFileError as e:
                    self.errorMessage(u'Error opening file',
                                      (u"<p><b>%s</b></p>"
                                       u"<p>Make sure <i>%s</i> is a valid label file.")
                                      % (e, unicodeFilePath))
                    self.status("Error reading %s" % unicodeFilePath)
                    return False
                self.imageData = self.labelFile.imageData
                self.lineColor = QColor(*self.labelFile.lineColor)
                self.fillColor = QColor(*self.labelFile.fillColor)
                self.canvas.verified = self.labelFile.verified
            else:
                # Load image:
                # read data first and store for saving into label file.
                self.imageData = read(unicodeFilePath, None)
                self.labelFile = None
                self.canvas.verified = False

            if isinstance(self.imageData, QImage):
                image = self.imageData
            else:
                image = QImage.fromData(self.imageData)
            if image.isNull():
                self.errorMessage(u'Error opening file',
                                  u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
                return False
            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            self.image = image
            self.filePath = unicodeFilePath
            self.canvas.loadPixmap(QPixmap.fromImage(image))
            if self.labelFile:
                self.loadLabels(self.labelFile.shapes)
            self.setClean()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self.filePath)
            self.toggleActions(True)
            self.showBoundingBoxFromAnnotationFile(self.filePath)

            self.setWindowTitle(__appname__ + ' ' + filePath)

            # Default : select last item if there is at least one item
            if self.labelList.count():
                self.labelList.setCurrentItem(self.labelList.item(self.labelList.count() - 1))
                self.labelList.item(self.labelList.count() - 1).setSelected(True)

            self.canvas.setFocus(True)
            if self.recommenderMode.isChecked() and not self.saved_label_exist:
                try:
                    self.box_recommender.detect(image_path=self.filePath)
                    self.drawPoints(self.box_recommender.points)
                except:
                    print("there isn't any recommended box or there is something wrong with the image")
            self.saved_label_exist = False
            return True
        return False

    def showBoundingBoxFromAnnotationFile(self, filePath):
        if self.defaultSaveDir is not None:
            basename = os.path.basename(os.path.splitext(filePath)[0])
            filedir = filePath.split(basename)[0].split(os.path.sep)[-2:-1][0]
            xmlPath = os.path.join(self.defaultSaveDir, basename + XML_EXT)
            txtPath = os.path.join(self.defaultSaveDir, basename + TXT_EXT)
            jsonPath = os.path.join(self.defaultSaveDir, filedir + JSON_EXT)
            csvPath = os.path.join(self.defaultSaveDir, basename + CSV_EXT)
            """Annotation file priority:
            PascalXML > YOLO
            """
            if os.path.isfile(self.csvFilePath):
                self.saved_label_exist = True
                self.loadshapesFromCsvFaceset()
            elif os.path.isfile(xmlPath):
                self.loadPascalXMLByFilename(xmlPath)
                self.saved_label_exist = True
            elif os.path.isfile(txtPath):
                self.loadYOLOTXTByFilename(txtPath)
                self.saved_label_exist = True
            elif os.path.isfile(jsonPath):
                self.loadCreateMLJSONByFilename(jsonPath, filePath)
                self.saved_label_exist = True
            elif os.path.isfile(csvPath):
                self.loadCsvByFilename(csvPath)
                self.saved_label_exist = True

        else:
            xmlPath = os.path.splitext(filePath)[0] + XML_EXT
            txtPath = os.path.splitext(filePath)[0] + TXT_EXT
            csvPath = os.path.splitext(filePath)[0] + CSV_EXT
            if os.path.isfile(self.csvFilePath):
                self.saved_label_exist = True
                self.loadshapesFromCsvFaceset()
            elif os.path.isfile(xmlPath):
                self.loadPascalXMLByFilename(xmlPath)
                self.saved_label_exist = True
            elif os.path.isfile(txtPath):
                self.loadYOLOTXTByFilename(txtPath)
                self.saved_label_exist = True
            elif os.path.isfile(csvPath):
                self.loadCsvByFilename(csvPath)
                self.saved_label_exist = True

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull() \
                and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.labelFontSize = int(0.02 * max(self.image.width(), self.image.height()))
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        settings = self.settings
        # If it loads images from dir, don't load it at the begining
        if self.dirname is None:
            settings[SETTING_FILENAME] = self.filePath if self.filePath else ''
        else:
            settings[SETTING_FILENAME] = ''

        settings[SETTING_WIN_SIZE] = self.size()
        settings[SETTING_WIN_POSE] = self.pos()
        settings[SETTING_WIN_STATE] = self.saveState()
        settings[SETTING_LINE_COLOR] = self.lineColor
        settings[SETTING_FILL_COLOR] = self.fillColor
        settings[SETTING_RECENT_FILES] = self.recentFiles
        settings[SETTING_ADVANCE_MODE] = not self._beginner
        if self.defaultSaveDir and os.path.exists(self.defaultSaveDir):
            settings[SETTING_SAVE_DIR] = ustr(self.defaultSaveDir)
        else:
            settings[SETTING_SAVE_DIR] = ''

        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            settings[SETTING_LAST_OPEN_DIR] = self.lastOpenDir
        else:
            settings[SETTING_LAST_OPEN_DIR] = ''
        if self.preProcessPath and os.path.isfile(self.preProcessPath):
            settings[SETTING_PREPROCESSING_PATH] = self.preProcessPath
        if self.prepathText.text() and os.path.isdir(self.prepathText.text()):
            settings[SETTING_PREPATH] = self.prepathText.text()
        if self.csvFilePath and os.path.isfile(self.csvFilePath):
            settings[SETTING_CSVDATABASE] = self.csvFilePath

        settings[SETTING_AUTO_SAVE] = self.autoSaving.isChecked()
        settings[SETTING_RECOMMENDER] = self.recommenderMode.isChecked()
        settings[SETTING_SINGLE_CLASS] = self.singleClassMode.isChecked()
        settings[SETTING_PAINT_LABEL] = self.displayLabelOption.isChecked()
        settings[SETTING_DRAW_SQUARE] = self.drawSquaresOption.isChecked()
        settings[SETTING_LABEL_FILE_FORMAT] = self.labelFileFormat
        settings.save()

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def scanAllImages(self, folderPath):
        extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        images = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.join(root, file)
                    path = ustr(os.path.abspath(relativePath))
                    images.append(path)
        natural_sort(images, key=lambda x: x.lower())
        return images

    def changeSavedirDialog(self, _value=False):
        if self.defaultSaveDir is not None:
            path = ustr(self.defaultSaveDir)
        else:
            path = '.'

        dirpath = ustr(QFileDialog.getExistingDirectory(self,
                                                        '%s - Save annotations to the directory' % __appname__, path,
                                                        QFileDialog.ShowDirsOnly
                                                        | QFileDialog.DontResolveSymlinks))

        if dirpath is not None and len(dirpath) > 1:
            self.defaultSaveDir = dirpath

        self.statusBar().showMessage('%s . Annotation will be saved to %s' %
                                     ('Change saved folder', self.defaultSaveDir))
        self.statusBar().show()

    def openAnnotationDialog(self, _value=False):
        if self.filePath is None:
            self.statusBar().showMessage('Please select image first')
            self.statusBar().show()
            return

        path = os.path.dirname(ustr(self.filePath)) \
            if self.filePath else '.'
        if self.labelFileFormat == LabelFileFormat.PASCAL_VOC:
            filters = "Open Annotation XML file (%s)" % ' '.join(['*.xml'])
            filename = ustr(QFileDialog.getOpenFileName(self, '%s - Choose a xml file' % __appname__, path, filters))
            if filename:
                if isinstance(filename, (tuple, list)):
                    filename = filename[0]
            self.loadPascalXMLByFilename(filename)

    def getprepath(self):
        defaultdir = ''
        if len(self.prepathText.text()) > 2:
            defaultdir = self.prepathText.text()

        targetDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                              '%s - Open Directory' % __appname__,
                                                              defaultdir,
                                                              QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        self.prepathText.setText(targetDirPath)

    def openDirDialog(self, _value=False, dirpath=None, silent=False):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else '.'
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filePath) if self.filePath else '.'
        if silent != True:
            targetDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                                  '%s - Open Directory' % __appname__,
                                                                  defaultOpenDirPath,
                                                                  QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        else:
            targetDirPath = ustr(defaultOpenDirPath)
        self.lastOpenDir = targetDirPath
        self.importDirImages(targetDirPath)

    def importDirImages(self, dirpath):
        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.dirname = dirpath
        self.filePath = None
        self.fileListWidget.clear()
        self.mImgList = self.scanAllImages(dirpath)
        self.openNextImg()
        for imgPath in self.mImgList:
            item = QListWidgetItem(imgPath)
            self.fileListWidget.addItem(item)

    def verifyImg(self, _value=False):
        # Proceding next image without dialog if having any label
        if self.filePath is not None:
            try:
                self.labelFile.toggleVerify()
            except AttributeError:
                # If the labelling file does not exist yet, create if and
                # re-save it with the verified attribute.
                self.saveFile()
                if self.labelFile != None:
                    self.labelFile.toggleVerify()
                else:
                    return

            self.canvas.verified = self.labelFile.verified
            self.paintCanvas()
            self.saveFile()

    def openPrevImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            return

        currIndex = self.mImgList.index(self.filePath)
        if currIndex - 1 >= 0:
            filename = self.mImgList[currIndex - 1]
            if filename:
                self.loadFile(filename)

    def openNextImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        filename = None
        if self.filePath is None:
            filename = self.mImgList[0]
        else:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex + 1 < len(self.mImgList):
                filename = self.mImgList[currIndex + 1]

        if filename:
            self.loadFile(filename)

    def getFacesetPath(self, ):
        path = os.path.dirname(ustr(self.filePath)) if self.filePath else '.'

        filename = QFileDialog.getOpenFileName(self, '%s - Choose your csv DB' % __appname__, path)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
        self.faceSetTextLine.setText(filename)
        self.csvFilePath = filename
        self.preProcessPath = filename

    def getPreProcessPath(self):

        if self.preProcessPath and os.path.exists(self.preProcessPath):
            defaultOpenDirPath = self.preProcessPath
        else:
            defaultOpenDirPath = os.path.dirname(self.filePath) if self.filePath else '.'

        targetDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                              '%s - choose dataset path' % __appname__,
                                                              defaultOpenDirPath,
                                                              QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        self.preProcessTextLine.setText(targetDirPath)
        self.preProcessPath = targetDirPath

    def getPreProcessCsvPath(self):
        path = os.path.dirname(ustr(self.preProcessPath)) if self.preProcessPath else '.'

        filename = QFileDialog.getOpenFileName(self, '%s - Choose your csv DB' % __appname__, path)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]

        self.preProcessTextLine.setText(filename)
        self.preProcessPath = filename

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(ustr(self.filePath)) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters = "Image & Label files (%s)" % ' '.join(formats + ['*%s' % LabelFile.suffix])
        filename = QFileDialog.getOpenFileName(self, '%s - Choose Image or Label file' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self.loadFile(filename)

    def saveFile(self, _value=False):
        if self.defaultSaveDir is not None and len(ustr(self.defaultSaveDir)):
            if self.filePath:
                imgFileName = os.path.basename(self.filePath)
                savedFileName = os.path.splitext(imgFileName)[0]
                savedPath = os.path.join(ustr(self.defaultSaveDir), savedFileName)
                self._saveFile(savedPath)
        else:
            imgFileDir = os.path.dirname(self.filePath)
            imgFileName = os.path.basename(self.filePath)
            savedFileName = os.path.splitext(imgFileName)[0]
            savedPath = os.path.join(imgFileDir, savedFileName)
            self._saveFile(savedPath if self.labelFile
                           else self.saveFileDialog(removeExt=False))

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self, removeExt=True):
        caption = '%s - Choose File' % __appname__
        filters = 'File (*%s)' % LabelFile.suffix
        openDialogPath = self.currentPath()
        dlg = QFileDialog(self, caption, openDialogPath, filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filenameWithoutExtension = os.path.splitext(self.filePath)[0]
        dlg.selectFile(filenameWithoutExtension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            fullFilePath = ustr(dlg.selectedFiles()[0])
            if removeExt:
                return os.path.splitext(fullFilePath)[0]  # Return file path without the extension.
            else:
                return fullFilePath
        return ''

    def _saveFile(self, annotationFilePath):
        if annotationFilePath and self.saveLabels(annotationFilePath):
            self.setClean()
            self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
            self.statusBar().show()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def deleteImg(self):
        deletePath = self.filePath
        if deletePath is not None:
            self.openNextImg()
            if os.path.exists(deletePath):
                os.remove(deletePath)
            self.importDirImages(self.lastOpenDir)

    def resetAll(self):
        self.settings.reset()
        self.close()
        proc = QProcess()
        proc.startDetached(os.path.abspath(__file__))

    def mayContinue(self):
        if not self.dirty:
            return True
        else:
            discardChanges = self.discardChangesDialog()
            if discardChanges == QMessageBox.No:
                return True
            elif discardChanges == QMessageBox.Yes:
                self.saveFile()
                return True
            else:
                return False

    def discardChangesDialog(self):
        yes, no, cancel = QMessageBox.Yes, QMessageBox.No, QMessageBox.Cancel
        msg = u'You have unsaved changes, would you like to save them and proceed?\nClick "No" to undo all changes.'
        return QMessageBox.warning(self, u'Attention', msg, yes | no | cancel)

    def sameNameExist(self, name):
        yes, no = QMessageBox.Yes, QMessageBox.No
        id = self.subject_name_to_id_dictionary[name]
        msg = u'there is {} in subjects with id = {} do you want save new id ? '.format(name, id)
        ans = QMessageBox.warning(self, u'Attention', msg, no | yes)
        if ans == yes:
            ans = QMessageBox.warning(self, u'Gender', u'male ?', yes | no)
            if ans == yes:
                return 1
            else:
                return 0
        return -1

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def chooseColor1(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            Shape.line_color = color
            self.canvas.setDrawingColor(color)
            self.canvas.update()
            self.setDirty()

    def deleteSelectedShape(self):
        self.remLabel(self.canvas.deleteSelected())
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def chshapeLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selectedShape.line_color = color
            self.canvas.update()
            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selectedShape.fill_color = color
            self.canvas.update()
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def loadPredefinedClasses(self, predefClassesFile):
        if os.path.exists(predefClassesFile) is True:
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.labelHist is None:
                        self.labelHist = [line]
                    else:
                        self.labelHist.append(line)

    def loadPascalXMLByFilename(self, xmlPath):
        if self.filePath is None:
            return
        if os.path.isfile(xmlPath) is False:
            return

        self.set_format(FORMAT_PASCALVOC)

        tVocParseReader = PascalVocReader(xmlPath)
        shapes = tVocParseReader.getShapes()
        self.loadLabels(shapes)
        self.canvas.verified = tVocParseReader.verified

    def loadYOLOTXTByFilename(self, txtPath):
        if self.filePath is None:
            return
        if os.path.isfile(txtPath) is False:
            return

        self.set_format(FORMAT_YOLO)
        tYoloParseReader = YoloReader(txtPath, self.image)
        shapes = tYoloParseReader.getShapes()
        self.loadLabels(shapes)
        self.canvas.verified = tYoloParseReader.verified

    def loadCreateMLJSONByFilename(self, jsonPath, filePath):
        if self.filePath is None:
            return
        if os.path.isfile(jsonPath) is False:
            return

        self.set_format(FORMAT_CREATEML)

        crmlParseReader = CreateMLReader(jsonPath, filePath)
        shapes = crmlParseReader.get_shapes()
        self.loadLabels(shapes)
        self.canvas.verified = crmlParseReader.verified

    def loadshapesFromCsvFaceset(self):
        if self.filePath is None:
            return

        shapes = getShapesFromCsvFaceSet(self.filePath, prepath=self.prepathText.text())
        if not shapes:
            self.saved_label_exist = False
            return
        for shape in shapes:
            self.drawPoints([(shape[0], shape[1]), (shape[2], shape[3])], shape[4], shape[5], drawingFlag=1,
                            isAuto=False)

    def loadCsvByFilename(self, csvPath):
        if self.filePath is None:
            return
        if os.path.isfile(csvPath) is False:
            return

        self.set_format(FORMAT_CSV)
        tCsvParseReader = CsvReader(csvPath, self.image)
        shapes = tCsvParseReader.getShapes()
        self.loadLabels(shapes)
        self.canvas.verified = tCsvParseReader.verified

    def copyPreviousBoundingBoxes(self):
        currIndex = self.mImgList.index(self.filePath)
        if currIndex - 1 >= 0:
            prevFilePath = self.mImgList[currIndex - 1]
            self.showBoundingBoxFromAnnotationFile(prevFilePath)
            self.saveFile()

    def togglePaintLabelsOption(self):
        for shape in self.canvas.shapes:
            shape.paintLabel = self.displayLabelOption.isChecked()

    def toogleDrawSquare(self):
        self.canvas.setDrawingShapeToSquare(self.drawSquaresOption.isChecked())

    def drawPoints(self, points, labelid=-1, mask=0, drawingFlag=0, isAuto=True):

        for i in range(len(points) // 2):
            p1x, p1y = points[2 * i]
            p2x, p2y = points[2 * i + 1]
            self.createShape()
            shape = Shape()
            p1 = QPointF(p1x, p1y)
            p2 = QPointF(p2x, p1y)
            p3 = QPointF(p2x, p2y)
            p4 = QPointF(p1x, p2y)
            shape.addPoint(p1)
            shape.addPoint(p2)
            shape.addPoint(p3)
            shape.addPoint(p4)
            shape.fill = True
            label = self.subject_dictionary[labelid]
            shape.label = label
            shape.userid = labelid
            if drawingFlag == 0:
                shape.recommendedPoints = [p1, p2, p3, p4]
                shape.userid = -1
            shape.mask = mask
            shape.drawingFlag = drawingFlag
            self.canvas.current = shape
            self.canvas.finalise()
            if not isAuto:
                self.setClean()

    def preProcessingWithPath(self):
        self.getPreProcessPath()
        path = self.preProcessPath

        if self.preProcessPath is None:
            self.errorMessage('None path', 'please choose a path ')
        elif not os.path.isdir(self.preProcessPath):
            self.errorMessage('wrong path', self.preProcessPath + "isn't directory")
        else:
            print('Getting embedings from : ' + path + ' please wait')
            if os.path.exists('embs.npy'):
                self.embsDict = np.load('embs.npy', allow_pickle='TRUE').item()
                self.len_faceset = len(list(self.embsDict.keys()))
            imagesPath = self.scanAllImages(path)
            for p in imagesPath:
                i = imagesPath.index(p)
                basename = os.path.basename(os.path.splitext(p)[0])
                filedir = p.split(basename)[0].split(os.path.sep)[-2:-1][0]
                if filedir == 'multi':
                    continue
                image = Image.open(p)
                image = np.array(image)
                try:
                    boxes, embs = self.box_recommender.detector.detectWithLandMark(image,
                                                                                   self.box_recommender.detector.detector)
                except:
                    continue
                if len(embs) == 0:
                    continue
                boxes = np.int16(np.round(boxes[0][0]))
                id = self.path_to_id_dictionary[p]
                id = 100 * id + 1
                self.embsDict[id] = [embs, boxes, filedir]
                stdout.write("\r" + str(i) + "/" + str(len(imagesPath)))
                stdout.flush()
            stdout.write("\n")
            print('successfully done!')
            np.save('embs.npy', self.embsDict)

    def write_embdings(self, csvPath, embspath, j=0):
        print('Getting embedings from : ' + csvPath + ' please wait')
        if os.path.exists(embspath):
            self.embsDict.update(np.load(embspath, allow_pickle='TRUE').item())
        with open(csvPath, mode='r', encoding='utf-8') as r:
            reader = csv.DictReader(r, oneCsvFile.FIELD_NAMES)
            id = j
            for row in reader:

                row_path = os.path.join(self.prepathText.text(), row['path'])

                if not os.path.isfile(row_path):
                    print('there is no image in given path : ' + row_path)
                    continue

                image = np.array(Image.open(row_path))

                shape = [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]

                img = cropImage(image, shape, thresh=10, )

                try:
                    boxes, embs = self.box_recommender.detector.detectWithLandMark(img,
                                                                                   self.box_recommender.detector.detector)
                except:
                    continue
                if len(embs) == 0:
                    continue
                shape = np.array(shape).astype(np.int16)

                self.embsDict[id] = [embs, shape, int(row['id'])]
                id += 1

        print('successfully done!')
        np.save(embspath, self.embsDict)
        return id

    def preProcessingWithCsv(self):
        self.getPreProcessCsvPath()
        csvPath = self.preProcessPath
        if csvPath == '':
            self.errorMessage('None path', 'please choose a path ')
        elif csvPath.split('.')[-1] != 'csv':
            self.errorMessage('wrong path', csvPath + "isn't csv")
        else:

            self.make_subject_dictionary()
            self.make_path_id_dictionary()
            lastId = self.write_embdings(csvPath, 'embs.npy')
            self.write_embdings('localfaceset.csv', 'localembs.npy', lastId)

    def read_paths(self, faceset_path, j=0):
        with open(faceset_path, mode='r', encoding='utf-8') as r:
            reader = csv.DictReader(r, oneCsvFile.FIELD_NAMES)
            i = j
            for row in reader:
                self.path_dictionary[i] = os.path.normpath(row['path'])
                i += 1
        return i

    def make_path_id_dictionary(self):
        localPath = 'localfaceset.csv'
        if not os.path.isfile(localPath):
            open(localPath, 'x')
        i = self.read_paths(self.preProcessPath)
        self.read_paths(localPath, i)
        self.path_to_id_dictionary = {self.path_dictionary[k]: k for k in self.path_dictionary}

    def update_subjects(self, newName):
        subjectPath = 'localsubjects.csv'
        ids = list(self.subject_dictionary.keys())
        names = list(self.subject_name_to_id_dictionary.keys())
        if newName in names:
            ans = self.sameNameExist(newName)
            if ans == -1:
                self.canvas.selectedShape.userid = self.subject_name_to_id_dictionary[newName]
                return
        else:
            ans = QMessageBox.warning(self, u'Gender', u'male ?', QMessageBox.Yes | QMessageBox.No)
        row = {}
        id = np.max(ids) + 1
        row['id'] = id
        row['name'] = newName
        row['gender'] = 'm' if ans == QMessageBox.Yes else 'f'
        self.canvas.selectedShape.userid = id
        with open(subjectPath, mode='a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, ['id', 'name', 'gender'])
            writer.writerow(row)
        self.make_subject_dictionary()

    def read_subjects(self, subjectPath):
        if not os.path.isfile:
            self.errorMessage('Subjects',
                              'please put the +' + os.path.basename(subjectPath) + 'on' + os.path.dirname(subjectPath))
            return
        with open(subjectPath, mode='r', encoding='utf-8', newline='') as r:
            reader = csv.DictReader(r, ['id', 'name'])
            for row in reader:
                try:
                    self.subject_dictionary[int(row['id'])] = row['name']
                except:
                    if row['id'] != 'id':
                        print('csvReader cant convert this row of subjects : ' + str(row))
                    continue

    def update_embeding(self):
        if os.path.exists('embs.npy'):
            a = np.load('embs.npy', allow_pickle='TRUE').item()
            self.len_faceset = len(list(a.keys()))
            self.embsDict.update(a)
        if os.path.exists('localembs.npy'):
            self.embsDict.update(np.load('localembs.npy', allow_pickle='TRUE').item())
        else:
            np.save('localembs', self.embsDict, )

    def make_subject_dictionary(self, ):
        dirPath = os.path.dirname(self.preProcessPath)
        subjectPath = dirPath + '/subjects.csv'
        localPath = 'localsubjects.csv'
        if not os.path.isfile(localPath):
            open(localPath, 'x')

        self.read_subjects(subjectPath)
        self.read_subjects(localPath)
        self.subject_dictionary[-1] = 'unknown'
        self.subject_dictionary[-2] = ''
        self.subject_name_to_id_dictionary = {self.subject_dictionary[k]: k for k in self.subject_dictionary}

    def showWindow(self, indexes):
        length = min(8, len(indexes))

        indexes.sort(key=lambda tup: tup[0], reverse=True)
        indexes = indexes[:length]

        for i in range(length):
            self.newFace(i, indexes)
        self.newInput(length)
        self.w.setWindowTitle("similarity recommends")
        self.w.show()

    def newFace(self, i, recomms, ):
        box = recomms[i][1]
        imgPath = recomms[i][2]
        try:
            image = Image.open(imgPath)

        except:
            return
        image = np.array(image)
        s = image.shape
        image = image[max(box[1] - 5, 0):min(box[3] + 5, s[1]), max(box[0] - 5, 0):min(box[2] + 5, s[1]), ::-1]
        image = cv2.resize(image, (self.image_size, self.image_size))
        cv2.imwrite('temp/' + str(i) + '.jpg', image)
        url = 'temp/' + str(i) + '.jpg'
        self.buttons.append(QPushButton(self.w))
        self.buttons[i].setStyleSheet(BUTTON_CSS.replace('xxxxx.jpg', url))
        self.buttons[i].setGeometry(i * self.image_size + self.image_pad, self.image_pad, self.image_size,
                                    self.image_size)
        self.buttons[i].clicked.connect(partial(self.p, recomms[i][3]))
        self.buttons[i].setToolTip(imgPath)
        self.buttons[i].setShortcut('Alt+' + str(i + 1))
        self.labels.append(
            QLabel(parent=self.w, text='name :' + self.subject_dictionary[recomms[i][3]] + '\nsimilarity : ' + str(
                round(recomms[i][0], 3))))

        self.labels[i].setGeometry(i * self.image_size + 3 * self.image_pad, self.image_size,
                                   self.image_size,
                                   10 * self.image_pad)

    def p(self, shapeid):
        name = self.subject_dictionary[shapeid]
        self.clickedItem = name
        self.canvas.selectedShape.label = name
        self.canvas.selectedShape.userid = shapeid
        self.canvas.selectedShape.mask = 1 * self.maskCheckBox.isChecked()
        item = self.currentItem()

        if self.clickedItem is None:
            text = item.text()
            if self.newName is not None:
                text = self.newName[0]
        else:
            text = self.clickedItem
        if text is not None:
            item.setText(text)
            item.setBackground(QColor(0, 255, 0, 200))
            self.setDirty()
            self.updateComboBox()

        self.w.close()

    def getNewOne(self):
        name = QInputDialog(self.w)
        self.newName = name.getText(self.w, 'title', 'Enter new Name :')
        item = self.currentItem()
        self.canvas.selectedShape.mask = 1 * self.maskCheckBox.isChecked()
        if self.clickedItem is None:
            text = item.text()
            if self.newName is not None:
                text = self.newName[0]
                self.update_subjects(text)
        else:
            text = self.clickedItem
        if text is not None:
            item.setText(text)
            item.setBackground(QColor(0, 255, 0, 200))
            self.setDirty()
            self.updateComboBox()
        self.w.close()

    def newInput(self, i):
        self.buttons.append(QPushButton(self.w))
        self.buttons[i].setText('new')
        self.buttons[i].setStyleSheet("background-color: \
                               rgba(255,255,0,255); \
                               color: rgba(0,0,0,255); \
                               border-style: solid; \
                               border-radius: 7px; border-width: 5px; \
                               border-color: rgba(0,0,0,255);")
        self.buttons[i].setGeometry(i * self.image_size + 3 * self.image_pad // 2,
                                    self.image_size // 2 + self.image_pad // 2,
                                    self.image_size - 6 * self.image_pad, 2 * self.image_pad)
        self.buttons[i].clicked.connect(self.getNewOne)
        self.buttons[i].setShortcut("Ctrl+n")


def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


def read(filename, default=None):
    try:
        reader = QImageReader(filename)
        reader.setAutoTransform(True)
        return reader.read()
    except:
        return default


def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    # Tzutalin 201705+: Accept extra agruments to change predefined class file
    argparser = argparse.ArgumentParser()
    argparser.add_argument("image_dir", nargs="?")
    argparser.add_argument("predefined_classes_file",
                           default=os.path.join(os.path.dirname(__file__), "data", "predefined_classes.txt"),
                           nargs="?")
    argparser.add_argument("save_dir", nargs="?")
    args = argparser.parse_args(argv[1:])
    # Usage : labelImg.py image predefClassFile saveDir
    win = MainWindow(args.image_dir,
                     args.predefined_classes_file,
                     args.save_dir)
    win.show()
    return app, win


def main():
    '''construct main app and run it'''
    if not os.path.exists('temp'):
        os.mkdir('temp')
    app, _win = get_main_app(sys.argv)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
