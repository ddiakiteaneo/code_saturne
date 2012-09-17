!-------------------------------------------------------------------------------

! This file is part of Code_Saturne, a general-purpose CFD tool.
!
! Copyright (C) 1998-2012 EDF S.A.
!
! This program is free software; you can redistribute it and/or modify it under
! the terms of the GNU General Public License as published by the Free Software
! Foundation; either version 2 of the License, or (at your option) any later
! version.
!
! This program is distributed in the hope that it will be useful, but WITHOUT
! ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
! FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
! details.
!
! You should have received a copy of the GNU General Public License along with
! this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
! Street, Fifth Floor, Boston, MA 02110-1301, USA.

!-------------------------------------------------------------------------------

subroutine modini
!================


!===============================================================================
! Purpose:
! --------

! Modify calculation parameters after user changes (module variables)

!-------------------------------------------------------------------------------
! Arguments
!__________________.____._____.________________________________________________.
! name             !type!mode ! role                                           !
!__________________!____!_____!________________________________________________!
!__________________!____!_____!________________________________________________!

!     Type: i (integer), r (real), s (string), a (array), l (logical),
!           and composite types (ex: ra real array)
!     mode: <-- input, --> output, <-> modifies data, --- work array
!===============================================================================

!===============================================================================
! Module files
!===============================================================================

use paramx
use cstnum
use dimens
use numvar
use optcal
use cstphy
use entsor
use albase
use radiat, only: iirayo
use alstru
use cplsat
use ppincl

!===============================================================================

implicit none

! Arguments


! Local variables

integer          ii, jj, ivar, iok, iest, imom, ikw
integer          icompt, ipp, nbccou, nn
integer          nscacp, iscal
double precision relxsp
double precision omgnrm, cosdto, sindto
double precision ux, uy, uz

!===============================================================================

! Indicateur erreur (0 : pas d'erreur)
iok = 0

!===============================================================================
! 1. ENTREES SORTIES entsor
!===============================================================================

! ---> Niveau d'impression listing
!       Non initialise -> standard
do ii = 1, nvarmx
  if (iwarni(ii).eq.-10000) then
    iwarni(ii) = 0
  endif
enddo

!---> Variables de calcul ITRSVR = ivar. Sinon 0 (valeur d'initialisation).

do ivar = 1, nvar
  itrsvr(ipprtp(ivar)) = ivar
enddo

!---> sorties chrono?
!     Sauf mention contraire de l'utilisateur, on sort a la fin les
!        variables de calcul, la viscosite, rho, le pas de temps s'il
!        est variable, les estimateurs s'ils sont actives, les moments
!        s'il y en a et la viscosite de maillage en ALE.

do ii = 2, nvppmx
  if (itrsvr(ii).ge.1.and.ichrvr(ii).eq.-999) then
    ichrvr(ii) = 1
  endif
enddo
ipp = ipppro(ipproc(irom))
if (ichrvr(ipp).eq.-999) ichrvr(ipp) = 1
ipp = ipppro(ipproc(ivisct))
if ((iturb.eq.10 .or. itytur.eq.2                 &
     .or. itytur.eq.5 .or. iturb.eq.60            &
     .or. iturb.eq.70)                            &
     .and.ichrvr(ipp).eq.-999) ichrvr(ipp) = 1
if (idtvar.lt.0) then
  ichrvr(ipppro(ipproc(icour))) = 0
  ichrvr(ipppro(ipproc(ifour))) = 0
endif
do iest = 1, nestmx
  if (iescal(iest).gt.0) then
    ipp = ipppro(ipproc(iestim(iest)))
    if (ichrvr(ipp).eq.-999) ichrvr(ipp) = 1
  endif
enddo
if (idtvar.eq.2.and.ichrvr(ippdt).eq.-999) ichrvr(ippdt) = 1
if (ipucou.ne.1) then
  ichrvr(ipptx) = 0
  ichrvr(ippty) = 0
  ichrvr(ipptz) = 0
endif

if (nbmomt.gt.0) then
  do imom = 1, nbmomt
    ipp = ipppro(ipproc(icmome(imom)))
    if (ichrvr(ipp).eq.-999) ichrvr(ipp) = 1
  enddo
endif
if (iale.eq.1) then
  call pstdfm
  !==========
  nn = 1
  if (iortvm.eq.1) nn = 3
  do ii = 1, nn
    ipp = ipppro(ipproc(ivisma(ii)))
    if (ichrvr(ipp).eq.-999) ichrvr(ipp) = 1
  enddo
endif

do ii = 1, nvppmx
  if (ichrvr(ii).eq.-999) then
    ichrvr(ii) = 0
  endif
enddo

icompt = 0
do ii = 2, nvppmx
  if (ichrvr(ii).eq.1) icompt = icompt+1
enddo

!---> sorties historiques ?
!      Si une valeur non modifiee par l'utilisateur (=-999)
!        on la met a sa valeur par defaut
!      On sort toutes les variables a tous les pas de temps par defaut
!      IHISVR nb de sonde et numero par variable (-999 non initialise)
!             -1 : toutes les sondes
!      NTHIST = -1 : on ne sort pas d'historiques
!      NTHIST =  n : on sort des historiques tous les n pas de temps
!      NTHSAV = -1 : on sauvegarde a la fin uniquement
!      NTHSAV =  0 : periode par defaut (voir caltri)
!             > 0  : periode

do ii = 2, nvppmx
  if (itrsvr(ii).ge.1.and.ihisvr(ii,1).eq.-999) then
    ihisvr(ii,1) = -1
  endif
enddo
if (ihisvr(ippdt, 1).eq.-999) ihisvr(ippdt, 1) = -1
if (ipucou.ne.1) then
  ihisvr(ipptx,1) = 0
  ihisvr(ippty,1) = 0
  ihisvr(ipptz,1) = 0
endif
ipp = ipppro(ipproc(ivisct))
if ((iturb.eq.10 .or. itytur.eq.2                        &
     .or. itytur.eq.5 .or. iturb.eq.60                   &
     .or. iturb.eq.70)                                   &
     .and.ihisvr(ipp,1).eq.-999) ihisvr(ipp,1) = -1
if (idtvar.lt.0) then
  ihisvr(ipppro(ipproc(icour)),1) = 0
  ihisvr(ipppro(ipproc(ifour)),1) = 0
endif
if (nbmomt.gt.0) then
  do imom = 1, nbmomt
    ipp = ipppro(ipproc(icmome(imom)))
    if (ihisvr(ipp,1).eq.-999) ihisvr(ipp,1) = -1
  enddo
endif

do ii = 1, nvppmx
  if (ihisvr(ii,1).eq.-999) then
    ihisvr(ii,1) = 0
  endif
enddo

!     Si on est en ALE, on a un test equivalent dans strini.F
if (iale.eq.0) then
  icompt = 0
  do ii = 2, nvppmx
    if (ihisvr(ii,1).ne.0) icompt = icompt+1
  enddo

  if (icompt.eq.0.or.ncapt.eq.0) then
    nthist = -1
    frhist = -1.d0
  endif
endif

! Adapt the output frequency parameters according to the time scheme.
if (idtvar.lt.0.or.idtvar.eq.2) then
  frhist = -1.d0
else
  if (frhist > 0.d0) then
    nthist = -1
  endif
endif

! ---> Nom des variables

if (nomvar(ipprtp(ipr)) .eq.' ') then
  nomvar(ipprtp(ipr)) = 'Pressure'
  if (icorio.eq.1) then
    nomvar(ipprtp(ipr)) = 'Rel Pressure'
  endif
endif
if (nomvar(ipprtp(iu)) .eq.' ') then
  nomvar(ipprtp(iu))   = 'VelocityX'
  if (icorio.eq.1) then
    nomvar(ipprtp(iu)) = 'Rel VelocityX'
  endif
endif
if (nomvar(ipprtp(iv)) .eq.' ') then
  nomvar(ipprtp(iv))   = 'VelocityY'
  if (icorio.eq.1) then
    nomvar(ipprtp(iv)) = 'Rel VelocityY'
  endif
endif
if (nomvar(ipprtp(iw)) .eq.' ') then
  nomvar(ipprtp(iw))   = 'VelocityZ'
  if (icorio.eq.1) then
    nomvar(ipprtp(iw)) = 'Rel VelocityZ'
  endif
endif
if (itytur.eq.2) then
  if (nomvar(ipprtp(ik)) .eq.' ') then
    nomvar(ipprtp(ik)) = 'Turb Kinetic Energy'
  endif
  if (nomvar(ipprtp(iep)) .eq.' ') then
    nomvar(ipprtp(iep)) = 'Turb Dissipation'
  endif
elseif (itytur.eq.3) then
  if (nomvar(ipprtp(ir11)) .eq.' ') then
    nomvar(ipprtp(ir11)) =  'R11'
  endif
  if (nomvar(ipprtp(ir22)) .eq.' ') then
    nomvar(ipprtp(ir22)) = 'R22'
  endif
  if (nomvar(ipprtp(ir33)) .eq.' ') then
    nomvar(ipprtp(ir33)) = 'R33'
  endif
  if (nomvar(ipprtp(ir12)) .eq.' ') then
    nomvar(ipprtp(ir12)) = 'R12'
  endif
  if (nomvar(ipprtp(ir13)) .eq.' ') then
    nomvar(ipprtp(ir13)) = 'R13'
  endif
  if (nomvar(ipprtp(ir23)) .eq.' ') then
    nomvar(ipprtp(ir23)) = 'R23'
  endif
  if (nomvar(ipprtp(iep)) .eq.' ') then
    nomvar(ipprtp(iep)) = 'Turb Dissipation'
  endif
  if (iturb.eq.32) then
    if (nomvar(ipprtp(ial)) .eq.' ') then
      nomvar(ipprtp(ial)) = 'Alphap'
    endif
  endif
elseif (itytur.eq.5) then
  if (nomvar(ipprtp(ik)) .eq.' ') then
    nomvar(ipprtp(ik)) = 'Turb Kinetic Energy'
  endif
  if (nomvar(ipprtp(iep)) .eq.' ') then
    nomvar(ipprtp(iep)) = 'Turb Dissipation'
  endif
  if (nomvar(ipprtp(iphi)) .eq.' ') then
    nomvar(ipprtp(iphi)) = 'Phi'
  endif
  if (iturb.eq.50) then
    if (nomvar(ipprtp(ifb)) .eq.' ') then
      nomvar(ipprtp(ifb)) = 'f_bar'
    endif
  elseif (iturb.eq.51) then
    if (nomvar(ipprtp(ial)) .eq.' ') then
      nomvar(ipprtp(ial)) = 'Alpha'
    endif
  endif
elseif (iturb.eq.60) then
  if (nomvar(ipprtp(ik)) .eq.' ') then
    nomvar(ipprtp(ik)) = 'Turb Kinetic Energy'
  endif
  if (nomvar(ipprtp(iomg)) .eq.' ') then
    nomvar(ipprtp(iomg)) = 'Omega'
  endif
elseif (iturb.eq.70) then
  if (nomvar(ipprtp(inusa)) .eq.' ') then
    nomvar(ipprtp(inusa)) = 'NuTilda'
  endif
endif

if (nomvar(ipppro(ipproc(irom))) .eq.' ') then
  nomvar(ipppro(ipproc(irom))) = 'Density'
endif
if (nomvar(ipppro(ipproc(ivisct))) .eq.' ') then
  nomvar(ipppro(ipproc(ivisct))) = 'Turb Viscosity'
endif
if (nomvar(ipppro(ipproc(iviscl))) .eq.' ') then
  nomvar(ipppro(ipproc(iviscl))) = 'Laminar Viscosity'
endif
if (ismago.gt.0) then
  if (nomvar(ipppro(ipproc(ismago))) .eq.' ') then
    nomvar(ipppro(ipproc(ismago))) = 'Csdyn2'
  endif
endif
if (icp.gt.0) then
  if (nomvar(ipppro(ipproc(icp))) .eq.' ') then
    nomvar(ipppro(ipproc(icp))) = 'Specific Heat'
  endif
endif
if (iescal(iespre).gt.0) then
  ipp = ipppro(ipproc(iestim(iespre)))
  if (nomvar(ipp) .eq.' ') then
    write(nomvar(ipp),'(a5,i1)') 'EsPre',iescal(iespre)
  endif
endif
if (iescal(iesder).gt.0) then
  ipp = ipppro(ipproc(iestim(iesder)))
  if (nomvar(ipp) .eq.' ') then
    write(nomvar(ipp),'(a5,i1)') 'EsDer',iescal(iesder)
  endif
endif
if (iescal(iescor).gt.0) then
  ipp = ipppro(ipproc(iestim(iescor)))
  if (nomvar(ipp) .eq.' ') then
    write(nomvar(ipp),'(a5,i1)') 'EsCor',iescal(iescor)
  endif
endif
if (iescal(iestot).gt.0) then
  ipp = ipppro(ipproc(iestim(iestot)))
  if (nomvar(ipp) .eq.' ') then
    write(nomvar(ipp),'(a5,i1)') 'EsTot',iescal(iestot)
  endif
endif

if (iscalt.gt.0.and.iscalt.le.nscal) then
  if (nomvar(ipprtp(isca(iscalt))) .eq.' ') then
    if (iscsth(iscalt).eq.2) then
      nomvar(ipprtp(isca(iscalt))) = 'Enthalpy'
    else
      if (iscalt.eq.ienerg) then
        nomvar(ipprtp(isca(iscalt))) = 'Total Energy'
      else if (iscalt.eq.itemp .or. itemp.lt.0) then
        nomvar(ipprtp(isca(iscalt))) = 'Temperature'
      endif
    endif
  endif
  if (nomvar(ipprtp(iut)) .eq.' ') then
    write(nomvar(ipprtp(iut)), '(a2)') 'ut'
  endif
  if (nomvar(ipprtp(ivt)) .eq.' ') then
    write(nomvar(ipprtp(ivt)), '(a2)') 'vt'
  endif
  if (nomvar(ipprtp(iwt)) .eq.' ') then
    write(nomvar(ipprtp(iwt)), '(a2)') 'wt'
  endif
endif

do jj = 1, nscaus
  ii = jj
  if (nomvar(ipprtp(isca(ii))) .eq.' ') then
    write(nomvar(ipprtp(isca(ii))),'(a5,i3.3)') 'Scaus', ii
  endif
enddo
do jj = 1, nscapp
  ii = iscapp(jj)
  if (nomvar(ipprtp(isca(ii))) .eq.' ') then
    write(nomvar(ipprtp(isca(ii))), '(a5,i3.3)') 'Scapp', ii
  endif
enddo

if (nbmomt.gt.0) then
  do imom = 1, nbmomt
    ipp = ipppro(ipproc(icmome(imom)))
    if (nomvar(ipp) .eq.' ') then
      write(nomvar(ipp), '(a6,i2.2)') 'MoyTps', imom
    endif
  enddo
endif

! total pressure (not defined in compressible case)
if (ippmod(icompf).lt.0) then
  ipp = ipppro(ipproc(iprtot))
  if (nomvar(ipp) .eq.' ') then
    nomvar(ipp)   = 'Total Pressure'
  endif
endif

ipp = ipppro(ipproc(icour))
if (nomvar(ipp) .eq.' ') then
  nomvar(ipp) = 'CFL'
endif

ipp = ipppro(ipproc(ifour))
if (nomvar(ipp) .eq.' ') then
  nomvar(ipp) = 'Fourier Number'
endif

if (nomvar(ippdt) .eq.' ') then
  nomvar(ippdt) = 'Local Time Step'
endif

if (nomvar(ipptx) .eq.' ') then
  nomvar(ipptx) = 'Tx'
endif
if (nomvar(ippty) .eq.' ') then
  nomvar(ippty) = 'Ty'
endif
if (nomvar(ipptz) .eq.' ') then
  nomvar(ipptz) = 'Tz'
endif

if (iale.eq.1) then
  if (nomvar(ipprtp(iuma)) .eq.' ') then
    nomvar(ipprtp(iuma)) = 'Mesh VelocityX'
  endif
  if (nomvar(ipprtp(ivma)) .eq.' ') then
    nomvar(ipprtp(ivma)) = 'Mesh VelocityY'
  endif
  if (nomvar(ipprtp(iwma)) .eq.' ') then
    nomvar(ipprtp(iwma)) = 'Mesh VelocityZ'
  endif
  if (iortvm.eq.0) then
    if (nomvar(ipppro(ipproc(ivisma(1)))) .eq.' ') then
      nomvar(ipppro(ipproc(ivisma(1)))) = 'Mesh ViscX'
    endif
  else
    if (nomvar(ipppro(ipproc(ivisma(1)))) .eq.' ') then
      nomvar(ipppro(ipproc(ivisma(1)))) = 'Mesh ViscX'
    endif
    if (nomvar(ipppro(ipproc(ivisma(2)))) .eq.' ') then
      nomvar(ipppro(ipproc(ivisma(2)))) = 'Mesh ViscY'
    endif
    if (nomvar(ipppro(ipproc(ivisma(3)))) .eq.' ') then
      nomvar(ipppro(ipproc(ivisma(3)))) = 'Mesh ViscZ'
    endif
  endif
endif

! ---> Sorties listing

ipp = ipppro(ipproc(irom))
if (irovar.eq.1.and.ilisvr(ipp).eq.-999) ilisvr(ipp) = 1
ipp = ipppro(ipproc(ivisct))
if ((iturb.eq.10 .or. itytur.eq.2                 &
     .or. itytur.eq.5 .or. iturb.eq.60            &
     .or. iturb.eq.70)                            &
     .and.ilisvr(ipp).eq.-999) ilisvr(ipp) = 1
if (inusa .gt. 0) then
  ipp = ipppro(ipproc(inusa))
  if (iturb.eq.70.and.ilisvr(ipp).eq.-999) ilisvr(ipp) = 1
endif
ipp = ipppro(ipproc(icour))
if (ilisvr(ipp).eq.-999 .or. idtvar.lt.0) ilisvr(ipp) = 0
ipp = ipppro(ipproc(ifour))
if (ilisvr(ipp).eq.-999 .or. idtvar.lt.0) ilisvr(ipp) = 0
if (iescal(iespre).gt.0) then
  ipp = ipppro(ipproc(iestim(iespre)))
  if (ilisvr(ipp).eq.-999) ilisvr(ipp) = 1
endif
if (iescal(iesder).gt.0) then
  ipp = ipppro(ipproc(iestim(iesder)))
  if (ilisvr(ipp).eq.-999) ilisvr(ipp) = 1
endif
if (iescal(iescor).gt.0) then
  ipp = ipppro(ipproc(iestim(iescor)))
  if (ilisvr(ipp).eq.-999) ilisvr(ipp) = 1
endif
if (iescal(iestot).gt.0) then
  ipp = ipppro(ipproc(iestim(iestot)))
  if (ilisvr(ipp).eq.-999) ilisvr(ipp) = 1
endif

if (nbmomt.gt.0) then
  do imom = 1, nbmomt
    ipp = ipppro(ipproc(icmome(imom)))
    if (ilisvr(ipp).eq.-999) ilisvr(ipp) = 1
  enddo
endif

if (ilisvr(ippdt).eq.-999) ilisvr(ippdt)  = 1
if (ipucou.ne.1 .or. idtvar.lt.0) then
  ilisvr(ipptx) = 0
  ilisvr(ippty) = 0
  ilisvr(ipptz) = 0
endif

do ii = 2, nvppmx
  if (itrsvr(ii).ge.1.and.ilisvr(ii).eq.-999) then
    ilisvr(ii) = 1
  endif
enddo
do ii = 1, nvppmx
  if (ilisvr(ii).eq.-999) then
    ilisvr(ii) = 0
  endif
enddo



!===============================================================================
! 2. POSITION DES VARIABLES DE numvar
!===============================================================================

! ---> Reperage des variables qui disposeront de deux types de CL

!     Fait dans varpos.
!     Si l'utilisateur y a touche ensuite, on risque l'incident.

!===============================================================================
! 3. OPTIONS DU CALCUL : TABLEAUX DE optcal
!===============================================================================

! ---> restart

call indsui(isuite)
!==========

! ---> Schema en temps

!   -- Flux de masse
if (abs(thetfl+999.d0).gt.epzero) then
  write(nfecra,1001) istmpf
  iok = iok + 1
elseif (istmpf.eq.0) then
  thetfl = 0.d0
elseif (istmpf.eq.2) then
  thetfl = 0.5d0
endif

!    -- Proprietes physiques
if (abs(thetro+999.d0).gt.epzero) then
  write(nfecra,1011) 'IROEXT',iroext,'THETRO'
  iok = iok + 1
elseif (iroext.eq.0) then
  thetro = 0.0d0
elseif (iroext.eq.1) then
  thetro = 0.5d0
elseif (iroext.eq.2) then
  thetro = 1.d0
endif
if (abs(thetvi+999.d0).gt.epzero) then
  write(nfecra,1011) 'IVIEXT',iviext,'THETVI'
  iok = iok + 1
elseif (iviext.eq.0) then
  thetvi = 0.0d0
elseif (iviext.eq.1) then
  thetvi = 0.5d0
elseif (iviext.eq.2) then
  thetvi = 1.d0
endif
if (abs(thetcp+999.d0).gt.epzero) then
  write(nfecra,1011) 'ICPEXT',icpext,'THETCP'
  iok = iok + 1
elseif (icpext.eq.0) then
  thetcp = 0.0d0
elseif (icpext.eq.1) then
  thetcp = 0.5d0
elseif (icpext.eq.2) then
  thetcp = 1.d0
endif

!    -- Termes sources NS
if (abs(thetsn+999.d0).gt.epzero) then
  write(nfecra,1011) 'ISNO2T',isno2t,'THETSN'
  iok = iok + 1
elseif (isno2t.eq.1) then
  thetsn = 0.5d0
elseif (isno2t.eq.2) then
  thetsn = 1.d0
elseif (isno2t.eq.0) then
  thetsn = 0.d0
endif

!    -- Termes sources grandeurs turbulentes
if (abs(thetst+999.d0).gt.epzero) then
  write(nfecra,1011) 'ISTO2T',isto2t,'THETST'
  iok = iok + 1
elseif (isto2t.eq.1) then
  thetst = 0.5d0
elseif (isto2t.eq.2) then
  thetst = 1.d0
elseif (isto2t.eq.0) then
  thetst = 0.d0
endif

do iscal = 1, nscal
!    -- Termes sources des scalaires
  if (abs(thetss(iscal)+999.d0).gt.epzero) then
    write(nfecra,1021) iscal,'ISSO2T',isso2t(iscal),'THETSS'
    iok = iok + 1
  elseif (isso2t(iscal).eq.1) then
    thetss(iscal) = 0.5d0
  elseif (isso2t(iscal).eq.2) then
    thetss(iscal) = 1.d0
  elseif (isso2t(iscal).eq.0) then
    thetss(iscal) = 0.d0
  endif
!    -- Diffusivite des scalaires
  if (abs(thetvs(iscal)+999.d0).gt.epzero) then
    write(nfecra,1021) iscal,'IVSEXT',ivsext(iscal),'THETVS'
    iok = iok + 1
  elseif (ivsext(iscal).eq.0) then
    thetvs(iscal) = 0.d0
  elseif (ivsext(iscal).eq.1) then
    thetvs(iscal) = 0.5d0
  elseif (ivsext(iscal).eq.2) then
    thetvs(iscal) = 1.d0
  endif
enddo

!     Ici on interdit que l'utilisateur fixe lui meme THETAV, par securite
!       mais on pourrait le laisser faire
!       (enlever le IOK, modifier le message et les tests dans verini)

!     Vitesse pression (la pression est prise sans interp)
if (abs(thetav(iu)+999.d0).gt.epzero.or.                 &
     abs(thetav(iv)+999.d0).gt.epzero.or.                 &
     abs(thetav(iw)+999.d0).gt.epzero.or.                 &
     abs(thetav(ipr)+999.d0).gt.epzero) then
  write(nfecra,1031) 'VITESSE-PRESSION ','THETAV'
  iok = iok + 1
elseif (ischtp.eq.1) then
  thetav(iu) = 1.d0
  thetav(iv) = 1.d0
  thetav(iw) = 1.d0
  thetav(ipr) = 1.d0
elseif (ischtp.eq.2) then
  thetav(iu) = 0.5d0
  thetav(iv) = 0.5d0
  thetav(iw) = 0.5d0
  thetav(ipr) = 1.d0
endif

!     Turbulence (en k-eps : ordre 1)
if (itytur.eq.2) then
  if (abs(thetav(ik )+999.d0).gt.epzero.or.               &
      abs(thetav(iep)+999.d0).gt.epzero) then
    write(nfecra,1031) 'VARIABLES   K-EPS','THETAV'
    iok = iok + 1
  elseif (ischtp.eq.1) then
    thetav(ik ) = 1.d0
    thetav(iep) = 1.d0
  elseif (ischtp.eq.2) then
    !     pour le moment, on ne peut pas passer par ici (cf varpos)
    thetav(ik ) = 0.5d0
    thetav(iep) = 0.5d0
  endif
elseif (itytur.eq.3) then
  if (abs(thetav(ir11)+999.d0).gt.epzero.or.              &
      abs(thetav(ir22)+999.d0).gt.epzero.or.              &
      abs(thetav(ir33)+999.d0).gt.epzero.or.              &
      abs(thetav(ir12)+999.d0).gt.epzero.or.              &
      abs(thetav(ir13)+999.d0).gt.epzero.or.              &
      abs(thetav(ir23)+999.d0).gt.epzero.or.              &
      abs(thetav(iep )+999.d0).gt.epzero) then
    write(nfecra,1031) 'VARIABLES  RIJ-EP','THETAV'
    iok = iok + 1
  elseif (ischtp.eq.1) then
    thetav(ir11) = 1.d0
    thetav(ir22) = 1.d0
    thetav(ir33) = 1.d0
    thetav(ir12) = 1.d0
    thetav(ir13) = 1.d0
    thetav(ir23) = 1.d0
    thetav(iep ) = 1.d0
  elseif (ischtp.eq.2) then
    thetav(ir11) = 0.5d0
    thetav(ir22) = 0.5d0
    thetav(ir33) = 0.5d0
    thetav(ir12) = 0.5d0
    thetav(ir13) = 0.5d0
    thetav(ir23) = 0.5d0
    thetav(iep ) = 0.5d0
  endif
  if (iturb.eq.32) then
    if (abs(thetav(ial)+999.d0).gt.epzero) then
      write(nfecra,1031) 'VARIABLES  RIJ-EB','THETAV'
      iok = iok + 1
    elseif (ischtp.eq.1) then
      thetav(ial) = 1.d0
    elseif (ischtp.eq.2) then
      thetav(ial) = 0.5d0
    endif
  endif

elseif (iturb.eq.50) then
  if (abs(thetav(ik  )+999.d0).gt.epzero.or.              &
      abs(thetav(iep )+999.d0).gt.epzero.or.              &
      abs(thetav(iphi)+999.d0).gt.epzero.or.              &
      abs(thetav(ifb )+999.d0).gt.epzero) then
    write(nfecra,1031) 'VARIABLES     V2F','THETAV'
    iok = iok + 1
  elseif (ischtp.eq.1) then
    thetav(ik  ) = 1.d0
    thetav(iep ) = 1.d0
    thetav(iphi) = 1.d0
    thetav(ifb ) = 1.d0
  elseif (ischtp.eq.2) then
    !     pour le moment, on ne peut pas passer par ici (cf varpos)
    thetav(ik  ) = 0.5d0
    thetav(iep ) = 0.5d0
    thetav(iphi) = 0.5d0
    thetav(ifb ) = 0.5d0
  endif
elseif (iturb.eq.51) then
  if (abs(thetav(ik  )+999.d0).gt.epzero.or.              &
      abs(thetav(iep )+999.d0).gt.epzero.or.              &
      abs(thetav(iphi)+999.d0).gt.epzero.or.              &
      abs(thetav(ial )+999.d0).gt.epzero) then
    write(nfecra,1031) 'VARIABLES BL-V2/K','THETAV'
    iok = iok + 1
  elseif (ischtp.eq.1) then
    thetav(ik  ) = 1.d0
    thetav(iep ) = 1.d0
    thetav(iphi) = 1.d0
    thetav(ial ) = 1.d0
  elseif (ischtp.eq.2) then
    !     pour le moment, on ne peut pas passer par ici (cf varpos)
    thetav(ik  ) = 0.5d0
    thetav(iep ) = 0.5d0
    thetav(iphi) = 0.5d0
    thetav(ial ) = 0.5d0
  endif
elseif (iturb.eq.60) then
  if (abs(thetav(ik  )+999.d0).gt.epzero.or.              &
      abs(thetav(iomg)+999.d0).gt.epzero ) then
    write(nfecra,1031) 'VARIABLES K-OMEGA','THETAV'
    iok = iok + 1
  elseif (ischtp.eq.1) then
    thetav(ik  ) = 1.d0
    thetav(iomg) = 1.d0
  elseif (ischtp.eq.2) then
    !     pour le moment, on ne peut pas passer par ici (cf varpos)
    thetav(ik  ) = 0.5d0
    thetav(iomg) = 0.5d0
  endif
elseif (iturb.eq.70) then
  if (abs(thetav(inusa)+999.d0).gt.epzero) then
    write(nfecra,1031) 'VARIABLE NU_tilde de SA','THETAV'
    iok = iok + 1
  elseif (ischtp.eq.1) then
    thetav(inusa) = 1.d0
  elseif (ischtp.eq.2) then
    !     pour le moment, on ne peut pas passer par ici (cf varpos)
    thetav(inusa) = 0.5d0
  endif
endif

!     Scalaires
do iscal = 1, nscal
  ivar  = isca(iscal)
  if (abs(thetav(ivar)+999.d0).gt.epzero) then
    write(nfecra,1041) 'SCALAIRE',ISCAL,'THETAV'
    iok = iok + 1
  elseif (ischtp.eq.1) then
    thetav(ivar) = 1.d0
  elseif (ischtp.eq.2) then
    thetav(ivar) = 0.5d0
  endif
  if ((iscal.eq.iscalt).and.(ityturt.eq.3)) then
    if (abs(thetav(iut)+999.d0).gt.epzero) then
      write(nfecra,1031) 'VARIABLE ut ','THETAV'
      iok = iok + 1
    elseif (ischtp.eq.1) then
      thetav(iut) = 1.d0
    elseif (ischtp.eq.2) then
      !     pour le moment, on ne peut pas passer par ici (cf varpos)
      thetav(iut) = 0.5d0
    endif
    if (abs(thetav(ivt)+999.d0).gt.epzero) then
      write(nfecra,1031) 'VARIABLE vt ','THETAV'
      iok = iok + 1
    elseif (ischtp.eq.1) then
      thetav(ivt) = 1.d0
    elseif (ischtp.eq.2) then
      !     pour le moment, on ne peut pas passer par ici (cf varpos)
      thetav(ivt) = 0.5d0
    endif
    if (abs(thetav(iwt)+999.d0).gt.epzero) then
      write(nfecra,1031) 'VARIABLE wt ','THETAV'
      iok = iok + 1
    elseif (ischtp.eq.1) then
      thetav(iwt) = 1.d0
    elseif (ischtp.eq.2) then
      !     pour le moment, on ne peut pas passer par ici (cf varpos)
      thetav(iwt) = 0.5d0
    endif
  endif
enddo

!     Vitesse de maillage en ALE
if (iale.eq.1) then
  if (abs(thetav(iuma)+999.d0).gt.epzero.or.                       &
      abs(thetav(ivma)+999.d0).gt.epzero.or.                       &
      abs(thetav(iwma)+999.d0).gt.epzero) then
    write(nfecra,1032) 'THETAV'
    iok = iok + 1
  elseif (ischtp.eq.1) then
    thetav(iuma) = 1.d0
    thetav(ivma) = 1.d0
    thetav(iwma) = 1.d0
  elseif (ischtp.eq.2) then
!     pour le moment, on ne peut pas passer par ici (cf varpos)
    thetav(iuma) = 0.5d0
    thetav(ivma) = 0.5d0
    thetav(iwma) = 0.5d0
  endif
endif

! ---> ISSTPC
!        Si l'utilisateur n'a rien specifie pour le test de pente (=-999),
!        On impose 1 (ie sans) pour la vitesse en LES
!                  0 (ie avec) sinon

if (itytur.eq.4) then
  ii = iu
  if (isstpc(ii).eq.-999) isstpc(ii) = 1
  ii = iv
  if (isstpc(ii).eq.-999) isstpc(ii) = 1
  ii = iw
  if (isstpc(ii).eq.-999) isstpc(ii) = 1
  do jj = 1, nscal
    ii = isca(jj)
    if (isstpc(ii).eq.-999) isstpc(ii) = 0
  enddo
endif

do ii = 1, nvarmx
  if (isstpc(ii).eq.-999) then
    isstpc(ii) = 0
  endif
enddo

! ---> BLENCV
!        Si l'utilisateur n'a rien specifie pour le schema convectif
!                  1 (ie centre) pour les vitesses et
!                                      les scalaires utilisateurs
!                  0 (ie upwind pur) pour le reste
!   (en particulier, en L.E.S. toutes les variables sont donc en centre)

ii = iu
if (abs(blencv(ii)+999.d0).lt.epzero) blencv(ii) = 1.d0
ii = iv
if (abs(blencv(ii)+999.d0).lt.epzero) blencv(ii) = 1.d0
ii = iw
if (abs(blencv(ii)+999.d0).lt.epzero) blencv(ii) = 1.d0
do jj = 1, nscaus
  ii = isca(jj)
  if (abs(blencv(ii)+999.d0).lt.epzero) blencv(ii) = 1.d0
enddo

do ii = 1, nvarmx
  if (abs(blencv(ii)+999.d0).lt.epzero) then
    blencv(ii) = 0.d0
  endif
enddo


! ---> NSWRSM, EPSRSM ET EPSILO
!        Si l'utilisateur n'a rien specifie  (NSWRSM=-999),
!        On impose
!           a l'ordre 1 :
!                  2  pour la pression
!                  1  pour les autres variables
!                  on initialise EPSILO a 1.D-8
!                  on initialise EPSRSM a 10*EPSILO
!           a l'ordre 2 :
!                  5  pour la pression
!                  10 pour les autres variables
!                  on initialise EPSILO a 1.D-5
!                  on initialise EPSRSM a 10*EPSILO
!     Attention aux tests dans verini

if (ischtp.eq.2) then
  ii = ipr
  if (nswrsm(ii).eq.-999) nswrsm(ii) = 5
  if (abs(epsilo(ii)+999.d0).lt.epzero) epsilo(ii) = 1.d-5
  if (abs(epsrsm(ii)+999.d0).lt.epzero) epsrsm(ii) = 10.d0*epsilo(ii)
  ii = iu
  if (nswrsm(ii).eq.-999) nswrsm(ii) = 10
  if (abs(epsilo(ii)+999.d0).lt.epzero) epsilo(ii) = 1.d-5
  if (abs(epsrsm(ii)+999.d0).lt.epzero) epsrsm(ii) = 10.d0*epsilo(ii)
  ii = iv
  if (nswrsm(ii).eq.-999) nswrsm(ii) = 10
  if (abs(epsilo(ii)+999.d0).lt.epzero) epsilo(ii) = 1.d-5
  if (abs(epsrsm(ii)+999.d0).lt.epzero) epsrsm(ii) = 10.d0*epsilo(ii)
  ii = iw
  if (nswrsm(ii).eq.-999) nswrsm(ii) = 10
  if (abs(epsilo(ii)+999.d0).lt.epzero) epsilo(ii) = 1.d-5
  if (abs(epsrsm(ii)+999.d0).lt.epzero) epsrsm(ii) = 10.d0*epsilo(ii)
  do jj = 1, nscal
    ii = isca(jj)
    if (nswrsm(ii).eq.-999) nswrsm(ii) = 10
    if (abs(epsilo(ii)+999.d0).lt.epzero) epsilo(ii) = 1.d-5
    if (abs(epsrsm(ii)+999.d0).lt.epzero) epsrsm(ii) = 10.d0*epsilo(ii)
  enddo
endif
ii = ipr
if (nswrsm(ii).eq.-999) nswrsm(ii) = 2

do ii = 1, nvarmx
  if (nswrsm(ii).eq.-999) nswrsm(ii) = 1
  if (abs(epsilo(ii)+999.d0).lt.epzero) epsilo(ii) = 1.d-8
  if (abs(epsrsm(ii)+999.d0).lt.epzero) epsrsm(ii) = 10.d0*epsilo(ii)
enddo

! ---> ANOMAX
!        Si l'utilisateur n'a rien specifie pour l'angle de non
!          orthogonalite pour la selection du voisinage etendu,
!          on impose pi/4 (utile aussi en mode verifications)

if (anomax.le.-grand) then
  anomax = pi*0.25d0
endif

! ---> IMLIGR
!        Si l'utilisateur n'a rien specifie pour la limitation des
!          gradients (=-999),
!        On impose -1 avec gradrc (pas de limitation)
!               et  1 avec gradmc (limitation)

if (imrgra.eq.0.or.imrgra.eq.4) then
  do ii = 1, nvarmx
    if (imligr(ii).eq.-999) then
      imligr(ii) = -1
    endif
  enddo
elseif (imrgra.eq.1.or.imrgra.eq.2.or.imrgra.eq.3) then
  do ii = 1, nvarmx
    if (imligr(ii).eq.-999) then
      imligr(ii) = 1
    endif
  enddo
endif

! ---> DTMIN DTMAX CDTVAR


if (dtmin.le.-grand) then
  dtmin = 0.1d0*dtref
endif
if (dtmax.le.-grand) then
  dtmax = 1.0d3*dtref
endif

!     Ici, ce n'est pas grave pour le moment,
!      etant entendu que ces coefs ne servent pas
!      s'ils servaient, attention dans le cas a plusieurs phases avec
!      une seule pression : celle ci prend le coef de la derniere phase
cdtvar(iv ) = cdtvar(iu)
cdtvar(iw ) = cdtvar(iu)
cdtvar(ipr) = cdtvar(iu)

if (itytur.eq.2) then
  cdtvar(iep ) = cdtvar(ik  )
elseif (itytur.eq.3) then
  cdtvar(ir22) = cdtvar(ir11)
  cdtvar(ir33) = cdtvar(ir11)
  cdtvar(ir12) = cdtvar(ir11)
  cdtvar(ir13) = cdtvar(ir11)
  cdtvar(ir23) = cdtvar(ir11)
  cdtvar(iep ) = cdtvar(ir11)
  ! cdtvar(ial) is useless because no time dependance in the equation of alpha.
  if (iturb.eq.32) then
    cdtvar(ial) = cdtvar(ir11)
  endif
elseif (itytur.eq.5) then
  cdtvar(iep ) = cdtvar(ik  )
  cdtvar(iphi) = cdtvar(ik  )
!     CDTVAR(IFB/IAL) est en fait inutile car pas de temps dans
!     l'eq de f_barre/alpha
  if (iturb.eq.50) then
    cdtvar(ifb ) = cdtvar(ik  )
  elseif (iturb.eq.51) then
    cdtvar(ial ) = cdtvar(ik  )
  endif
elseif (iturb.eq.60) then
  cdtvar(iomg) = cdtvar(ik  )
elseif (iturb.eq.70) then
  ! cdtvar est � 1.0 par defaut dans iniini.f90
  cdtvar(inusa)= cdtvar(inusa)
endif
if ((nscaus.gt.0).and.(ityturt.eq.3)) then
  cdtvar(iut) = cdtvar(iu)
  cdtvar(ivt) = cdtvar(iu)
  cdtvar(iwt) = cdtvar(iu)
endif

! ---> IDEUCH, YPLULI
!      En laminaire, longueur de melange, Spalar-Allmaras et LES,
!      une echelle de vitesse.
!      Sinon, 2 echelles, sauf si l'utilisateur choisit 1 echelle.
!      On a initialise IDEUCH a -999 pour voir si l'utilisateur essaye
!        de choisir deux echelles quand ce n'est pas possible et le
!        prevenir dans la section verification.

if (ideuch.eq.-999) then
  if (iturb.eq. 0.or.                                     &
      iturb.eq.10.or.                                     &
      itytur.eq.4.or.                                     &
      iturb.eq.70) then
    ideuch = 0
  else
    ideuch = 1
  endif
endif

! Pour YPLULI, 1/XKAPPA est la valeur qui assure la continuite de la derivee
! entre la zone lineaire et la zone logarithmique.

! Dans le cas des lois de paroi invariantes, on utilise la valeur de
! continuite du profil de vitesse, 10.88.

! Pour la LES, on remet 10.88, afin d'eviter des clic/clac quand on est a
! la limite (en modele a une echelle en effet, YPLULI=1/XKAPPA ne permet pas
! forcement de calculer u* de maniere totalement satisfaisante).
! Idem en Spalart-Allmaras.

if (ypluli.lt.-grand) then
  if (ideuch.eq.2 .or. itytur.eq.4 .or. iturb.eq.70) then
    ypluli = 10.88d0
  else
    ypluli = 1.d0/xkappa
  endif
endif


! ---> Van Driest
if (idries.eq.-1) then
  !   On met 1 en supposant qu'en periodicite ou parallele on utilise le
  !     mode de calcul de la distance a la paroi qui les prend en charge
  !     (ICDPAR=+/-1, valeur par defaut)
  if (iturb.eq.40) then
    idries = 1
  elseif (iturb.eq.41) then
    idries = 0
  elseif (iturb.eq.42) then
    idries = 0
  endif
endif


! ---> ICPSYR
!      Si l'utilisateur n'a pas modifie ICPSYR, on prend par defaut :
!        s'il n y a pas de couplage
!          0 pour tous les scalaires
!        sinon
!          1 pour le scalaire ISCALT s'il existe
!          0 pour les autres
!      Les modifs adequates devront etre ajoutees pour les physiques
!        particulieres
!      Les tests de coherence seront faits dans verini.

if (nscal.gt.0) then

!     On regarde s'il y a du couplage

  call nbcsyr (nbccou)
  !==========

!     S'il y a du couplage
  if (nbccou .ne. 0) then

!       On compte le nombre de scalaires couples
    nscacp = 0
    do iscal = 1, nscal
      if (icpsyr(iscal).eq.1) then
        nscacp = nscacp + 1
      endif
    enddo

!       Si l'utilisateur n'a pas couple de scalaire,
    if (nscacp.eq.0) then

!         On couple le scalaire temperature de la phase
      if (iscalt.gt.0.and.iscalt.le.nscal) then
        icpsyr(iscalt) = 1
        goto 100
      endif
 100        continue

    endif

  endif

!     Pour tous les autres scalaires, non renseignes pas l'utilisateur
!       on ne couple pas
  do iscal = 1, nscamx
    if (icpsyr(iscal).eq.-999) then
      icpsyr(iscal) = 0
    endif
  enddo

endif


! ---> ISCSTH
!      Si l'utilisateur n'a pas modifie ISCSTH, on prend par defaut :
!        scalaire passif  pour les scalaires autres que ISCALT
!      Les modifs adequates devront etre ajoutees pour les physiques
!        particulieres
!      Noter en outre que, par defaut, si on choisit temperature
!        elle est en K (ceci n'est utile que pour le rayonnement et les pp)

!         =-10: non renseigne
!         =-1 : temperature en C
!         = 0 : passif
!         = 1 : temperature en K
!         = 2 : enthalpie



if (nscal.gt.0) then
  do ii = 1, nscal
    if (iscsth(ii).eq.-10)then
      if (ii.ne.iscalt) then
        iscsth(ii) = 0
      endif
    endif
  enddo
endif

! ---> ICALHY
!      Calcul de la pression hydrostatique en sortie pour les conditions de
!        Dirichlet sur la pression. Se deduit de IPHYDR et de la valeur de
!        la gravite (test assez arbitraire sur la norme).
!      ICALHY est initialise a -1 (l'utilisateur peut avoir force
!        0 ou 1 et dans ce cas, on ne retouche pas)

if (icalhy.ne.-1.and.icalhy.ne.0.and.icalhy.ne.1) then
  write(nfecra,1061)icalhy
  iok = iok + 1
endif


! ---> IDGMOM
!      Calcul du degre des moments

do imom = 1, nbmomx
  idgmom(imom) = 0
enddo
do imom = 1, nbmomt
  do ii = 1, ndgmox
    if (idfmom(ii,imom).ne.0) idgmom(imom) = idgmom(imom) + 1
  enddo
enddo

! ---> ICDPAR
!      Calcul de la distance a la paroi. En standard, on met ICDPAR a -1, au cas
!      ou les faces de bord auraient change de type d'un calcul a l'autre. En k-omega,
!      il faut la distance a la paroi pour une suite propre, donc on initialise a 1 et
!      on avertit (dans verini).
ikw = 0
if (iturb.eq.60) ikw = 1
if (icdpar.eq.-999) then
  icdpar = -1
  if (ikw.eq.1) icdpar = 1
  if (isuite.eq.1 .and. ikw.eq.1) write(nfecra,2000)
endif
if (icdpar.eq.-1 .and. ikw.eq.1 .and. isuite.eq.1)                &
     write(nfecra,2001)

! ---> INEEDY, IMLIGY
!      Calcul de la distance a la paroi
!       (une seule phase ...)

ineedy = 0
if ((iturb.eq.30.and.irijec.eq.1).or.              &
     (itytur.eq.4.and.idries.eq.1).or.              &
     iturb.eq.60.or.iturb.eq.70      ) then
  ineedy = 1
endif

if (imrgra.eq.0 .or. imrgra.eq.4) then
  if (imligy.eq.-999) then
    imligy = -1
  endif
elseif (imrgra.eq.1.or.imrgra.eq.2.or.imrgra.eq.3) then
  if (imligy.eq.-999) then
    imligy = 1
  endif
endif

!     Warning : non initialise => comme la vitesse
if (iwarny.eq.-999) then
  iwarny = iwarni(iu)
endif


! ---> IKECOU
!     En k-eps prod lin, v2f ou k-omega, on met IKECOU a 0 par defaut,
!     sinon on le laisse a 1
!     Dans verini on bloquera le v2f et le k-eps prod lin si IKECOU.NE.0
!     On bloquera aussi le stationnaire si IKECOU.NE.0
if (ikecou.eq.-999) then
  if (idtvar.lt.0) then
    ikecou = 0
  else if (iturb.eq.21 .or. itytur.eq.5           &
       .or. iturb.eq.60 ) then
    ikecou = 0
  else
    ikecou = 1
  endif
endif

! ---> RELAXV
if (idtvar.lt.0) then
  relxsp = 1.d0-relxst
  if (relxsp.le.epzero) relxsp = relxst
  if (abs(relaxv(ipr)+999.d0).le.epzero)                 &
       relaxv(ipr) = relxsp
  do ii = 1, nvarmx
    if (abs(relaxv(ii)+999.d0).le.epzero) relaxv(ii) = relxst
  enddo
else
  if (ikecou.eq.0) then
    if (itytur.eq.2 .or. itytur.eq.5) then
      if (abs(relaxv(ik)+999.d0).lt.epzero)              &
           relaxv(ik) = 0.7d0
      if (abs(relaxv(iep)+999.d0).lt.epzero)             &
           relaxv(iep) = 0.7d0
    else if (itytur.eq.6) then
      if (abs(relaxv(ik)+999.d0).lt.epzero)              &
           relaxv(ik) = 0.7d0
      if (abs(relaxv(iomg)+999.d0).lt.epzero)            &
           relaxv(iomg) = 0.7d0
    endif
  endif
  if (iturb.eq.70) then
    if (abs(relaxv(inusa)+999.d0).lt.epzero) then
      relaxv(inusa) = 1.D0
    endif
  endif
  if (abs(relaxv(ipr)+999.d0).lt.epzero)                 &
       relaxv(ipr) = 1.d0
endif

! ---> SPECIFIQUE STATIONNAIRE
if (idtvar.lt.0) then
  dtref = 1.d0
  dtmin = 1.d0
  dtmax = 1.d0
  do ii = 1, nvarmx
    istat(ii) = 0
  enddo
  arak = arak/max(relaxv(iu),epzero)
endif

! ---> INEEDF
!     Si on a demande un posttraitement des efforts aux bords, on
!     les calcule !
if (mod(ipstdv,ipstfo).eq.0) then
  ineedf = 1
endif
!     Si on est en ALE, par defaut on calcule les efforts aux bords
!     (un test eventuel sur la presence de structures viendrait trop
!     tard)
if (iale.eq.1) ineedf = 1

!===============================================================================
! 4. TABLEAUX DE cstphy
!===============================================================================

! ---> Constantes
!    Ca fait un calcul en double, mais si qqn a bouge cmu, apow, bpow,
!     ca servira.

cpow    = apow**(2.d0/(1.d0-bpow))
dpow    = 1.d0/(1.d0+bpow)
cmu025 = cmu**0.25d0

! ---> ICLVFL
!      Si l'utilisateur n'a pas modifie ICLVFL, on prend par defaut :
!        0 pour les variances
!      Les modifs adequates devront etre ajoutees pour les physiques
!        particulieres

do iscal = 1, nscal
  if (iscavr(iscal).gt.0) then
    if (iclvfl(iscal).eq.-1) then
      iclvfl(iscal) = 0
    endif
  endif
enddo


! ---> VISLS0 (IVISLS ont ete verifies dans varpos)

! For scalars which are not variances, define the default diffusivity

! Pour les variances de fluctuations, les valeurs du tableau
! precedent ne doivent pas avoir ete modifiees par l'utilisateur
! Elles sont prises egales aux valeurs correspondantes pour le
! scalaire associe.

if (nscaus.gt.0) then
  do jj = 1, nscaus
    if (iscavr(jj).le.0 .and. visls0(jj).lt.-grand) then
      visls0(jj) = viscl0
    endif
  enddo
endif

if (nscal.gt.0) then
  do ii = 1, nscal
    iscal = iscavr(ii)
    if (iscal.gt.0.and.iscal.le.nscal)then
      if (visls0(ii).lt.-grand) then
        visls0(ii) = visls0(iscal)
      else
        write(nfecra,1071)ii,                                     &
          ii,iscal,ii,iscal,                                      &
          ii,ii,iscal,visls0(iscal)
        iok = iok + 1
      endif
    endif
  enddo
endif

! ---> XYZP0 : reference pour la pression hydrostatique
!      On considere que l'utilisateur a specifie la reference
!      a partir du moment ou il a specifie une coordonnee.
!      Pour les coordonnees non specifiees, on met 0.

do ii = 1, 3
  if (xyzp0(ii).gt.-0.5d0*rinfin) then
    ixyzp0 = 1
  else
    xyzp0(ii) = 0.d0
  endif
enddo

! Turbulent fluxes constant for GGDH, AFM and DFM
if (nscal.gt.0) then
  if (iturbt.eq.0) then
    do iscal = 1, nscal
      idften(isca(iscal)) = 1
    enddo

  ! AFM and GGDH on the thermal scalar
  elseif (ityturt.eq.1.or.ityturt.eq.2) then
    if (iscalt.gt.0) then
      idften(isca(iscalt)) = 6
      ctheta(iscalt) = cthafm
    else
      call csexit(1)
    endif

  ! DFM on the thermal scalar
  elseif (ityturt.eq.3) then
    if (iscalt.gt.0) then
      idifft(isca(iscalt)) = 0
      idften(isca(iscalt)) = 1
      ctheta(iscalt) = cthdfm
    else
      call csexit(1)
    endif
    ! GGDH on the thermal fluxes
    idften(iut) = 6
    idften(ivt) = 6
    idften(iwt) = 6
    ! GGDH on the variance of the thermal scalar
    do iscal = 1, nscal
      if (iscavr(iscal).eq.iscalt) then
        idften(isca(iscal)) = 6
        ctheta(iscal) = csrij
      endif
    enddo

  endif
endif

! Vecteur rotation et matrice(s) associees

omgnrm = sqrt(omegax**2 + omegay**2 + omegaz**2)

if (omgnrm.ge.epzero) then

  ! Normalized rotation vector

  ux = omegax / omgnrm
  uy = omegay / omgnrm
  uz = omegaz / omgnrm

  ! Matrice de projection sur l'axe de rotation

  prot(1,1) = ux**2
  prot(2,2) = uy**2
  prot(3,3) = uz**2

  prot(1,2) = ux*uy
  prot(2,1) = prot(1,2)

  prot(1,3) = ux*uz
  prot(3,1) = prot(1,3)

  prot(2,3) = uy*uz
  prot(3,2) = prot(2,3)

  ! Antisymetrc representation of Omega

  qrot(1,1) = 0.d0
  qrot(2,2) = 0.d0
  qrot(3,3) = 0.d0

  qrot(1,2) = -uz
  qrot(2,1) = -qrot(1,2)

  qrot(1,3) =  uy
  qrot(3,1) = -qrot(1,3)

  qrot(2,3) = -ux
  qrot(3,2) = -qrot(2,3)

  ! Matrice de rotation

  cosdto = cos(dtref*omgnrm)
  sindto = sin(dtref*omgnrm)

  do ii = 1, 3
    do jj = 1, 3
      irot(ii,jj) = 0.d0
    enddo
    irot(ii,ii) = 1.d0
  enddo

  do ii = 1, 3
    do jj = 1, 3
      rrot(ii,jj) = cosdto*irot(ii,jj) + (1.d0 - cosdto)*prot(ii,jj) &
                                       +         sindto *qrot(ii,jj)
    enddo
  enddo

else

  do ii = 1, 3
    do jj = 1, 3
      irot(ii,jj) = 0.d0
      prot(ii,jj) = 0.d0
      qrot(ii,jj) = 0.d0
      rrot(ii,jj) = 0.d0
    enddo
    irot(ii,ii) = 1.d0
    rrot(ii,ii) = 1.d0
  enddo

endif


!===============================================================================
! 5. ELEMENTS DE albase
!===============================================================================

if (iale.eq.1) then
  if (isuite.eq.0 .and. italin.eq.-999 ) italin = 1
else
  italin = 0
endif

!===============================================================================
! 6. COEFFICIENTS DE alstru
!===============================================================================

if (betnmk.lt.-0.5d0*grand) betnmk = (1.d0-alpnmk)**2/4.d0
if (gamnmk.lt.-0.5d0*grand) gamnmk = (1.d0-2.d0*alpnmk)/2.d0
if (aexxst.lt.-0.5d0*grand) aexxst = 0.5d0
if (bexxst.lt.-0.5d0*grand) bexxst = 0.0d0
if (cfopre.lt.-0.5d0*grand) cfopre = 2.0d0

!===============================================================================
! 7. PARAMETRES DE cplsat
!===============================================================================

! Get coupling number

call nbccpl(nbrcpl)
!==========

if (nbrcpl.ge.1) then
  ! Si on est en couplage rotor/stator avec resolution en repere absolu
  omgnrm = sqrt(omegax**2 + omegay**2 + omegaz**2)
  if (omgnrm.ge.epzero) then
    ! Couplage avec interpolation aux faces
    ifaccp = 1
    ! Maillage mobile
    if (icorio.eq.0) then
      imobil = 1
      call pstdfm
      !==========
    endif
  endif
endif

!===============================================================================
! 8. STOP SI PB
!===============================================================================

if (iok.ne.0) then
  call csexit (1)
endif

#if defined(_CS_LANG_FR)

 1001 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION : ARRET A L''ENTREE DES DONNEES',               /,&
'@    =========,'                                               /,&
'@    ISTMPF = ',   i10,                                        /,&
'@    THETFL SERA INITIALISE AUTOMATIQUEMENT.,'                 /,&
'@    NE PAS LE MODIFIER.,'                                     /,&
'@',                                                            /,&
'@  Le calcul ne sera pas execute.',                            /,&
'@',                                                            /,&
'@  Verifier cs_user_parameters.f90,'                           /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 1011 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION : ARRET A L''ENTREE DES DONNEES',               /,&
'@    =========,'                                               /,&
'@    ',a6,' = ',   i10,                                        /,&
'@    ',a6,' SERA INITIALISE AUTOMATIQUEMENT.,'                 /,&
'@    NE PAS LE MODIFIER.,'                                     /,&
'@',                                                            /,&
'@  Le calcul ne sera pas execute.',                            /,&
'@',                                                            /,&
'@  Verifier cs_user_parameters.f90,'                           /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 1021 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION : ARRET A L''ENTREE DES DONNEES',               /,&
'@    =========,'                                               /,&
'@    SCALAIRE ',   i10,' ',a6,' = ',   i10,                    /,&
'@    ',a6,' SERA INITIALISE AUTOMATIQUEMENT.,'                 /,&
'@    NE PAS LE MODIFIER.,'                                     /,&
'@',                                                            /,&
'@  Le calcul ne sera pas execute.',                            /,&
'@',                                                            /,&
'@  Verifier cs_user_parameters.f90,'                           /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 1031 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION : ARRET A L''ENTREE DES DONNEES',               /,&
'@    =========,'                                               /,&
'@    ',a17,                                                    /,&
'@    ',a6,' SERA INITIALISE AUTOMATIQUEMENT.,'                 /,&
'@    NE PAS LE MODIFIER.,'                                     /,&
'@',                                                            /,&
'@  Le calcul ne sera pas execute.',                            /,&
'@',                                                            /,&
'@  Verifier cs_user_parameters.f90,'                           /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 1032 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION : ARRET A L''ENTREE DES DONNEES',               /,&
'@    =========,'                                               /,&
'@    VITESSE DE MAILLAGE EN ALE',                              /,&
'@    ',a6,' SERA INITIALISE AUTOMATIQUEMENT.,'                 /,&
'@    NE PAS LE MODIFIER.,'                                     /,&
'@',                                                            /,&
'@  Le calcul ne sera pas execute.',                            /,&
'@',                                                            /,&
'@  Verifier cs_user_parameters.f90,'                           /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 1041 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION : ARRET A L''ENTREE DES DONNEES',               /,&
'@    =========,'                                               /,&
'@    ',a8,' ',i10,                                             /,&
'@    ',a6,' SERA INITIALISE AUTOMATIQUEMENT.,'                 /,&
'@    NE PAS LE MODIFIER.,'                                     /,&
'@',                                                            /,&
'@  Le calcul ne sera pas execute.',                            /,&
'@',                                                            /,&
'@  Verifier cs_user_parameters.f90,'                           /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 1061 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION : ARRET A L''ENTREE DES DONNEES',               /,&
'@    =========,'                                               /,&
'@    ICALHY doit etre un entier egal a 0 ou 1',                /,&
'@',                                                            /,&
'@  Il vaut ici ',i10,                                          /,&
'@',                                                            /,&
'@  Le calcul ne sera pas execute.',                            /,&
'@',                                                            /,&
'@  Verifier cs_user_parameters.f90,'                           /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 1071 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION : ARRET A L''ENTREE DES DONNEES',               /,&
'@    =========,'                                               /,&
'@    SCALAIRE ',i10,   ' NE PAS MODIFIER LA DIFFUSIVITE',      /,&
'@',                                                            /,&
'@  Le scalaire ',i10,   ' represente la variance des,'         /,&
'@    fluctuations du scalaire ',i10,                           /,&
'@                             (ISCAVR(',i10,   ') = ',i10,     /,&
'@  La diffusivite VISLS0(',i10,   ') du scalaire ',i10,        /,&
'@    ne doit pas etre renseignee :,'                           /,&
'@    elle sera automatiquement prise egale a la diffusivite,'  /,&
'@    du scalaire ',i10,   ' soit ',e14.5,                      /,&
'@',                                                            /,&
'@  Le calcul ne sera pas execute.',                            /,&
'@',                                                            /,&
'@  Verifier cs_user_parameters.f90,'                           /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 2000 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION :       A L''ENTREE DES DONNEES',               /,&
'@    =========,'                                               /,&
'@',                                                            /,&
'@  Le modele de turbulence k-omega a ete choisi. Pour gerer,'  /,&
'@    correctement la suite de calcul, l''indicateur ICDPAR a', /,&
'@    ete mis a 1 (relecture de la distance a la paroi dans le',/,&
'@    fichier suite).',                                         /,&
'@  Si cette initialisation pose probleme (modification du,'    /,&
'@    nombre et de la position des faces de paroi depuis le',   /,&
'@    calcul precedent), forcer ICDPAR=-1 (il peut alors',      /,&
'@    y avoir un leger decalage dans la viscosite',             /,&
'@    turbulente au premier pas de temps).',                    /,&
'@',                                                            /,&
'@  Le calcul sera execute.',                                   /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 2001 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION :       A L''ENTREE DES DONNEES',               /,&
'@    =========,'                                               /,&
'@',                                                            /,&
'@  Le modele de turbulence k-omega a ete choisi, avec',        /,&
'@    l''option de recalcul de la distance a la paroi,'         /,&
'@    (ICDPAR=-1). Il se peut qu''il y ait un leger decalage,'  /,&
'@    dans la viscosite turbulente au premier pas de temps.',   /,&
'@',                                                            /,&
'@  Le calcul sera execute.',                                   /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)

#else

 1001 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING: ABORT IN THE DATA SPECIFICATION',                /,&
'@    ========',                                                /,&
'@    ISTMPF = ',   i10,                                        /,&
'@    THETFL WILL BE AUTOMATICALLY INITIALIZED.',               /,&
'@    DO NOT MODIFY IT.,'                                       /,&
'@',                                                            /,&
'@  The calculation will not be run.',                          /,&
'@',                                                            /,&
'@  Check cs_user_parameters.f90',                              /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 1011 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING: ABORT IN THE DATA SPECIFICATION',                /,&
'@    ========',                                                /,&
'@    ',a6,' = ',   i10,                                        /,&
'@    ',a6,' WILL BE INITIALIZED AUTOMATICALLY',                /,&
'@    DO NOT MODIFY IT.,'                                       /,&
'@',                                                            /,&
'@  The calculation will not be run.',                          /,&
'@',                                                            /,&
'@  Check cs_user_parameters.f90',                              /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 1021 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING: ABORT IN THE DATA SPECIFICATION',                /,&
'@    ========',                                                /,&
'@    SCALAR ',   i10,' ',a6,' = ',   i10,                      /,&
'@    ',a6,' WILL BE INITIALIZED AUTOMATICALLY',                /,&
'@    DO NOT MODIFY IT.,'                                       /,&
'@',                                                            /,&
'@  The calculation will not be run.',                          /,&
'@',                                                            /,&
'@  Check cs_user_parameters.f90',                              /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 1031 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING: ABORT IN THE DATA SPECIFICATION',                /,&
'@    ========',                                                /,&
'@    ',a17,                                                    /,&
'@    ',a6,' WILL BE INITIALIZED AUTOMATICALLY',                /,&
'@    DO NOT MODIFY IT.,'                                       /,&
'@',                                                            /,&
'@  The calculation will not be run.',                          /,&
'@',                                                            /,&
'@  Check cs_user_parameters.f90',                              /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 1032 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING: ABORT IN THE DATA SPECIFICATION',                /,&
'@    ========',                                                /,&
'@    MESH VELOCITY IN ALE',                                    /,&
'@    ',a6,' WILL BE INITIALIZED AUTOMATICALLY',                /,&
'@    DO NOT MODIFY IT.,'                                       /,&
'@',                                                            /,&
'@  The calculation will not be run.',                          /,&
'@',                                                            /,&
'@  Check cs_user_parameters.f90',                              /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 1041 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING: ABORT IN THE DATA SPECIFICATION',                /,&
'@    ========',                                                /,&
'@    ',a8,' ',i10,                                             /,&
'@    ',a6,' WILL BE INITIALIZED AUTOMATICALLY',                /,&
'@    DO NOT MODIFY IT.,'                                       /,&
'@',                                                            /,&
'@  The calculation will not be run.',                          /,&
'@',                                                            /,&
'@  Check cs_user_parameters.f90',                              /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 1061 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING: ABORT IN THE DATA SPECIFICATION',                /,&
'@    ========',                                                /,&
'@    ICALHY must be an integer equal to 0 or 1',               /,&
'@',                                                            /,&
'@  Its value is ',i10,                                         /,&
'@',                                                            /,&
'@  The calculation will not be run.',                          /,&
'@',                                                            /,&
'@  Check cs_user_parameters.f90',                              /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 1071 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING: ABORT IN THE DATA SPECIFICATION',                /,&
'@    ========',                                                /,&
'@    SCALAR ',i10,   ' DO NOT MODIFY THE DIFFUSIVITY,'         /,&
'@',                                                            /,&
'@  The scalar ',i10,   ' is the fluctuations variance',        /,&
'@    of the scalar ',i10,                                      /,&
'@                             (ISCAVR(',i10,   ') = ',i10,     /,&
'@  The diffusivity VISLS0(',i10,   ') of the scalar ',i10,     /,&
'@    must not be set:',                                        /,&
'@    it will be automatically set equal to the scalar',        /,&
'@    diffusivity ',i10,   ' i.e. ',e14.5,                      /,&
'@',                                                            /,&
'@  The calculation will not be run.',                          /,&
'@',                                                            /,&
'@  Check cs_user_parameters.f90',                              /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 2000 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING:       IN THE DATA SPECIFICATION',                /,&
'@    ========',                                                /,&
'@',                                                            /,&
'@  The k-omega turbulence model has been chosen. In order to', /,&
'@    have a correct calculation restart, the ICDPAR indicator',/,&
'@    has been set to 1 (read the wall distance in the restart',/,&
'@    file).',                                                  /,&
'@  If this initialization raises any issue (modification of,'  /,&
'@    the number and position of the wall faces since the',     /,&
'@    previous calcuation), force ICDPAR=1 (there might be,'    /,&
'@    a small shift in the turbulent viscosity at the,'         /,&
'@    first time-step).,'                                       /,&
'@',                                                            /,&
'@  The calculation will be run.',                              /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 2001 format(                                                     &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING:       IN THE DATA SPECIFICATION',                /,&
'@    ========',                                                /,&
'@',                                                            /,&
'@  The k-omega turbulence model has been chosen, with the,'    /,&
'@    option for a re-calculation of the wall distance',        /,&
'@    (ICDPAR=-1). There might be a small shift in the',        /,&
'@    turbulent viscosity at the first time-step.',             /,&
'@',                                                            /,&
'@  The calculation will be run.',                              /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)

#endif

return
end subroutine
