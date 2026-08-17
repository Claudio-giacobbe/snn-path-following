sim = require('sim')

-- ============================================================
-- GENERATORE DI PERCORSI CHIUSI CASUALI
-- CoppeliaSim 4.10
--
-- CHILD SCRIPT di /Floor
-- Lua / Simulation script / Non-threaded
--
-- FUNZIONAMENTO:
-- 1) nasconde il vecchio /Floor/path
-- 2) crea un percorso ROSSO CHIUSO
-- 3) il percorso parte dalla posizione del robot
-- 4) il robot fa un giro completo
-- 5) quando torna al punto di partenza viene generato
--    automaticamente un NUOVO percorso
--
-- Il /Floor/path originale NON viene cancellato.
-- ============================================================


local floor = -1
local robot = -1
local originalPath = -1

local generatedSegments = {}

local startX = 0
local startY = 0

local hasLeftStart = false
local lastGenerationTime = -100


-- ============================================================
-- FUNZIONI UTILI
-- ============================================================

local function clamp(v, a, b)
    return math.max(a, math.min(b, v))
end


local function distance2D(x1, y1, x2, y2)

    local dx = x1 - x2
    local dy = y1 - y2

    return math.sqrt(
        dx * dx + dy * dy
    )
end


-- Elimina SOLO il percorso generato dal nostro script.
local function clearGeneratedPath()

    for i = #generatedSegments, 1, -1 do

        local h = generatedSegments[i]

        if h ~= nil and sim.isHandle(h) then
            sim.removeObject(h)
        end

    end

    generatedSegments = {}

end


-- ============================================================
-- GENERAZIONE DI UN LOOP
-- ============================================================

local function generateNewPath()

    clearGeneratedPath()

    -- --------------------------------------------------------
    -- POSIZIONE ATTUALE DEL ROBOT
    -- --------------------------------------------------------

    local robotPos =
        sim.getObjectPosition(
            robot,
            floor
        )

    local robotOri =
        sim.getObjectOrientation(
            robot,
            floor
        )

    local rx0 = robotPos[1]
    local ry0 = robotPos[2]

    -- Direzione frontale del robot.
    local heading = robotOri[3]

    -- Vettore davanti al robot.
    local ux = math.cos(heading)
    local uy = math.sin(heading)

    -- Vettore laterale.
    local vx = -math.sin(heading)
    local vy = math.cos(heading)


    -- --------------------------------------------------------
    -- DIMENSIONI REALI DEL FLOOR
    -- --------------------------------------------------------

    local floorSize, floorBB =
        sim.getShapeBB(floor)

    local halfX =
        floorSize[1] * 0.5 - 0.35

    local halfY =
        floorSize[2] * 0.5 - 0.35

    local floorTop =
        floorBB[3]
        +
        floorSize[3] * 0.5

    local pathZ =
        floorTop + 0.012


    -- --------------------------------------------------------
    -- CERCA UNA ELLISSE CHE PASSI ESATTAMENTE DAL ROBOT
    -- --------------------------------------------------------
    --
    -- Il punto iniziale della curva ?:
    --
    --             START
    --               *
    --               |
    --             LOOP
    --
    -- La tangente iniziale viene allineata alla direzione
    -- del robot.
    -- --------------------------------------------------------

    local bestRx = nil
    local bestRy = nil
    local bestCx = nil
    local bestCy = nil

    -- Proviamo diverse dimensioni.
    for attempt = 1, 200 do

        -- LOOP QUASI GRANDE QUANTO TUTTA LA SCACCHIERA.
        --
        -- Partiamo da circa l'88% della met? del Floor e
        -- cerchiamo la curva pi? grande che rimane dentro
        -- i bordi.
        local rx =
            halfX * (0.80 + math.random() * 0.12)

        local ry =
            halfY * (0.80 + math.random() * 0.12)

        -- Centro: il robot ? il punto superiore/laterale
        -- dell'ellisse.
        --
        -- local point at theta=0 = (0, ry)
        --
        local cx =
            rx0 - vx * ry

        local cy =
            ry0 - vy * ry

        local valid = true

        -- Controlliamo 48 punti dell'ellisse.
        for k = 0, 47 do

            local theta =
                2.0 * math.pi
                *
                k / 48.0

            local lx =
                rx * math.sin(theta)

            local ly =
                ry * math.cos(theta)

            local px =
                cx
                +
                ux * lx
                +
                vx * ly

            local py =
                cy
                +
                uy * lx
                +
                vy * ly

            if
                px < -halfX
                or px > halfX
                or
                py < -halfY
                or py > halfY
            then

                valid = false
                break

            end

        end

        if valid then

            bestRx = rx
            bestRy = ry
            bestCx = cx
            bestCy = cy

            break

        end

    end


    -- --------------------------------------------------------
    -- FALLBACK
    -- --------------------------------------------------------

    if bestRx == nil then

        -- Loop piccolo di sicurezza.
        bestRx = halfX * 0.82
        bestRy = halfY * 0.82

        bestCx =
            rx0 - vx * bestRy

        bestCy =
            ry0 - vy * bestRy

    end


    -- --------------------------------------------------------
    -- CREA I PUNTI DEL LOOP
    -- --------------------------------------------------------
    --
    -- Il loop occupa quasi tutta la scacchiera, ma NON ? una
    -- semplice ellisse:
    --
    --  * 6 armoniche casuali deformano il bordo
    --  * la deformazione ? continua e morbida
    --  * il punto iniziale rimane esattamente sotto il robot
    --  * niente angoli secchi
    --
    -- In questo modo ogni giro ha curve diverse.
    -- --------------------------------------------------------

    local numberOfPoints = 96

    -- Coefficienti casuali ma moderati.
    -- Le armoniche basse producono curve ampie;
    -- quelle alte aggiungono variazioni pi? piccole.
    local radial = {}
    local tangential = {}

    for h = 1, 6 do

        radial[h] =
            -0.10
            +
            math.random() * 0.20

        tangential[h] =
            -0.08
            +
            math.random() * 0.16

    end


    local points = {}


    for i = 0, numberOfPoints - 1 do

        local theta =
            2.0 * math.pi
            *
            i / numberOfPoints


        -- ----------------------------------------------------
        -- DEFORMAZIONE RADIALE
        -- ----------------------------------------------------

        local radialFactor = 1.0

        for h = 1, 6 do

            radialFactor =
                radialFactor
                +
                radial[h]
                *
                math.sin(h * theta)

        end


        -- ----------------------------------------------------
        -- DEFORMAZIONE TANGENZIALE
        -- ----------------------------------------------------

        local tangentFactor = 0.0

        for h = 1, 5 do

            tangentFactor =
                tangentFactor
                +
                tangential[h]
                *
                math.cos(h * theta)

        end


        -- ----------------------------------------------------
        -- ELLISSE DI BASE
        -- ----------------------------------------------------

        local lx =
            bestRx
            *
            radialFactor
            *
            math.sin(theta)

        local ly =
            bestRy
            *
            radialFactor
            *
            math.cos(theta)


        -- Movimento laterale che crea curve non simmetriche.
        local lateral =
            bestRy
            *
            tangentFactor
            *
            math.sin(theta)


        ly = ly + lateral


        -- ----------------------------------------------------
        -- COORDINATE NEL FRAME DEL ROBOT
        -- ----------------------------------------------------

        local px =
            bestCx
            +
            ux * lx
            +
            vx * ly

        local py =
            bestCy
            +
            uy * lx
            +
            vy * ly


        -- ----------------------------------------------------
        -- IL PRIMO PUNTO ? ESATTAMENTE IL ROBOT
        -- ----------------------------------------------------

        if i == 0 then

            px = rx0
            py = ry0

        end


        points[#points + 1] = {
            px,
            py
        }

    end


    -- --------------------------------------------------------
    -- CORREZIONE DEI BORDI
    -- --------------------------------------------------------
    --
    -- Se una deformazione casuale porta un punto troppo vicino
    -- al bordo, lo riportiamo dentro.
    -- --------------------------------------------------------

    for i = 1, #points do

        points[i][1] =
            clamp(
                points[i][1],
                -halfX + 0.12,
                halfX - 0.12
            )

        points[i][2] =
            clamp(
                points[i][2],
                -halfY + 0.12,
                halfY - 0.12
            )

    end


    -- Ripristina ESATTAMENTE lo start.
    points[1][1] = rx0
    points[1][2] = ry0


    -- --------------------------------------------------------
    -- CREA I SEGMENTI ROSSI
    -- --------------------------------------------------------

    -- LARGHEZZA DELLA STRISCIA ROSSA.
    -- Aumentata ulteriormente per essere facilmente visibile
    -- dalla camera del robot.
    local pathWidth = 0.28
    local pathHeight = 0.012


    for i = 1, #points do

        -- IMPORTANTE:
        -- ultimo punto -> primo punto
        --
        -- Questo rende il percorso CHIUSO.

        local nextIndex = i + 1

        if nextIndex > #points then
            nextIndex = 1
        end


        local p1 = points[i]
        local p2 = points[nextIndex]


        local dx =
            p2[1] - p1[1]

        local dy =
            p2[2] - p1[2]


        local length =
            math.sqrt(
                dx * dx
                +
                dy * dy
            )


        if length > 0.001 then

            local angle =
                math.atan2(
                    dy,
                    dx
                )


            -- Sovrapposizione minima per evitare buchi.
            local segmentLength =
                length + 0.025


            local midX =
                (p1[1] + p2[1])
                * 0.5

            local midY =
                (p1[2] + p2[2])
                * 0.5


            local shape =
                sim.createPrimitiveShape(
                    sim.primitiveshape_cuboid,
                    {
                        segmentLength,
                        pathWidth,
                        pathHeight
                    },
                    0
                )


            if shape ~= -1 then

                -- Figlio del Floor.
                sim.setObjectParent(
                    shape,
                    floor,
                    false
                )


                sim.setObjectPosition(
                    shape,
                    {
                        midX,
                        midY,
                        pathZ
                    },
                    floor
                )


                sim.setObjectOrientation(
                    shape,
                    {
                        0,
                        0,
                        angle
                    },
                    floor
                )


                -- Statico.
                sim.setObjectInt32Param(
                    shape,
                    sim.shapeintparam_static,
                    1
                )


                -- Non respondable.
                sim.setObjectInt32Param(
                    shape,
                    sim.shapeintparam_respondable,
                    0
                )


                -- ROSSO.
                sim.setShapeColor(
                    shape,
                    nil,
                    sim.colorcomponent_ambient_diffuse,
                    {
                        1,
                        0,
                        0
                    }
                )


                generatedSegments[
                    #generatedSegments + 1
                ] = shape

            end

        end

    end


    -- --------------------------------------------------------
    -- NUOVO PUNTO DI PARTENZA
    -- --------------------------------------------------------

    startX = rx0
    startY = ry0

    hasLeftStart = false

    lastGenerationTime =
        sim.getSimulationTime()


    -- --------------------------------------------------------
    -- DEBUG
    -- --------------------------------------------------------

    print('')
    print('==============================================')
    print(' NUOVO PERCORSO CHIUSO GENERATO')
    print('==============================================')
    print(
        'Punti: '
        ..
        #points
    )
    print(
        'Segmenti: '
        ..
        #generatedSegments
    )
    print(
        string.format(
            'Dimensioni loop: %.2f x %.2f m',
            bestRx * 2,
            bestRy * 2
        )
    )
    print(
        string.format(
            'Floor disponibile: %.2f x %.2f m',
            halfX * 2,
            halfY * 2
        )
    )
    print(
        'Start: '
        ..
        string.format(
            'X=%.2f Y=%.2f',
            startX,
            startY
        )
    )
    print(
        'Percorso CHIUSO: ultimo segmento -> primo segmento'
    )
    print(
        'Curve: deformazione casuale a 6 armoniche'
    )
    print(
        'Il vecchio /Floor/path NON viene cancellato.'
    )
    print('==============================================')
    print('')

end


-- ============================================================
-- INIT
-- ============================================================

function sysCall_init()

    math.randomseed(
        os.time()
        +
        math.floor(
            sim.getSystemTimeInMs(-1)
        )
    )


    print('')
    print('==============================================')
    print(' GENERATORE LOOP RANDOM')
    print('==============================================')


    -- OGGETTI REALI DELLA SCENA.
    -- Usiamo i percorsi assoluti perch? sappiamo che nella
    -- tua scena esistono realmente.

    floor =
        sim.getObject('/Floor')

    robot =
        sim.getObject('/BM_Bot')

    originalPath =
        sim.getObject('/Floor/path')


    print(
        'Floor handle: '
        ..
        floor
    )

    print(
        'BM_Bot handle: '
        ..
        robot
    )

    print(
        'Path originale handle: '
        ..
        originalPath
    )


    -- --------------------------------------------------------
    -- NASCONDI IL PATH ORIGINALE
    -- --------------------------------------------------------

    sim.setObjectInt32Param(
        originalPath,
        sim.objintparam_visibility_layer,
        0
    )


    -- --------------------------------------------------------
    -- CREA IL PRIMO LOOP
    -- --------------------------------------------------------

    generateNewPath()

end


-- ============================================================
-- CONTROLLO DEL GIRO COMPLETO
-- ============================================================

function sysCall_sensing()

    if robot == -1 then
        return
    end


    local pos =
        sim.getObjectPosition(
            robot,
            floor
        )


    local x = pos[1]
    local y = pos[2]


    local d =
        distance2D(
            x,
            y,
            startX,
            startY
        )


    -- --------------------------------------------------------
    -- IL ROBOT DEVE PRIMA ALLONTANARSI DALLO START
    -- --------------------------------------------------------

    if not hasLeftStart then

        if d > 0.80 then

            hasLeftStart = true

            print(
                'Il robot ha lasciato lo start: '
                ..
                string.format(
                    'd=%.2f m',
                    d
                )
            )

        end

        return

    end


    -- --------------------------------------------------------
    -- ? TORNATO ALLO START?
    -- --------------------------------------------------------

    local currentTime =
        sim.getSimulationTime()


    -- Evitiamo di rigenerare pi? volte nello stesso punto.
    if
        currentTime
        -
        lastGenerationTime
        <
        3.0
    then

        return

    end


    -- Quando torna entro 35 cm dal punto di partenza,
    -- consideriamo completato il giro.
    if d < 0.35 then

        print('')
        print('==============================================')
        print(' GIRO COMPLETATO!')
        print(' Generazione nuovo percorso...')
        print('==============================================')
        print('')


        generateNewPath()

    end

end


-- ============================================================
-- CLEANUP
-- ============================================================

function sysCall_cleanup()

    -- Elimina solo i segmenti generati.
    clearGeneratedPath()


    -- Ripristina il percorso originale.
    if
        originalPath ~= -1
        and
        sim.isHandle(originalPath)
    then

        sim.setObjectInt32Param(
            originalPath,
            sim.objintparam_visibility_layer,
            1
        )

    end

end