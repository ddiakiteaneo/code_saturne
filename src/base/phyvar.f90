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

subroutine phyvar &
!================

 ( nvar   , nscal  ,                                              &
   dt     , rtp    , rtpa   , propce , propfa , propfb ,          &
   coefa  , coefb  )

!===============================================================================
! FONCTION :
! --------

! REMPLISSAGE DES GRANDEURS PHYSIQUES VARIABLES EN TEMPS
!    ESSENTIELLEMENT LA VISCOSITE TURBULENTE.
!    ON APPELLE UN SOUS PROGRAMME UTILISATEUR QUI PERMET DE
!    SPECIFIER ROM, ROMB, VISCL, VISCLS ...

! Arguments
!__________________.____._____.________________________________________________.
! name             !type!mode ! role                                           !
!__________________!____!_____!________________________________________________!
! nvar             ! i  ! <-- ! total number of variables                      !
! nscal            ! i  ! <-- ! total number of scalars                        !
! dt(ncelet)       ! ra ! <-- ! time step (per cell)                           !
! rtp, rtpa        ! ra ! <-- ! calculated variables at cell centers           !
!  (ncelet, *)     !    !     !  (at current and previous time steps)          !
! propce(ncelet, *)! ra ! <-- ! physical properties at cell centers            !
! propfa(nfac, *)  ! ra ! <-- ! physical properties at interior face centers   !
! propfb(nfabor, *)! ra ! <-- ! physical properties at boundary face centers   !
! coefa, coefb     ! ra ! <-- ! boundary conditions                            !
!  (nfabor, *)     !    !     !                                                !
!__________________!____!_____!________________________________________________!

!     TYPE : E (ENTIER), R (REEL), A (ALPHANUMERIQUE), T (TABLEAU)
!            L (LOGIQUE)   .. ET TYPES COMPOSES (EX : TR TABLEAU REEL)
!     MODE : <-- donnee, --> resultat, <-> Donnee modifiee
!            --- tableau de travail
!===============================================================================

!===============================================================================
! Module files
!===============================================================================

use paramx
use numvar
use optcal
use cstphy
use cstnum
use entsor
use pointe
use albase
use parall
use period
use ihmpre
use ppppar
use ppthch
use ppincl
use mesh

!===============================================================================

implicit none

! Arguments

integer          nvar   , nscal


double precision dt(ncelet), rtp(ncelet,*), rtpa(ncelet,*)
double precision propce(ncelet,*)
double precision propfa(nfac,*), propfb(nfabor,*)
double precision coefa(nfabor,*), coefb(nfabor,*)

! Local variables

character*80     chaine
integer          ivar  , iel   , ifac  , iscal
integer          ii    , jj    , iok   , iok1  , iok2  , iisct
integer          nn
integer          ibrom , ipcrom, ipbrom, ipcvst
integer          ipccp , ipcvis, ipcvma
integer          ivarh
integer          iclipc
integer          nswrgp, imligp, iwarnp, iphydp, iclvar
integer          ir12ip, ir13ip, ir23ip, ialpip,iccocg,inc
Double precision xk, xe, xnu, xrom, vismax(nscamx), vismin(nscamx)
double precision nusa, xi3, fv1, cv13
double precision varmn(4), varmx(4), tt, ttmin, ttke, viscto, xrtp
double precision alp3, xrij(3,3) , xnal(3)   , xnoral
double precision xttke, xttkmg, xttdrb,epsrgp, climgp, extrap
double precision alpha, ym, yk
double precision trrij, csteps

integer          ipass
data             ipass /0/
save             ipass

!===============================================================================

!===============================================================================
! 1.  INITIALISATIONS
!===============================================================================

ipass = ipass + 1

!===============================================================================
! 2.  PREPARATION DE LA PERIODICITE DE ROTATION
!       CALCUL DE DUDXYZ ET DRDXYZ (gradients sur les halos avec prise
!       en compte des periodicites pour exploitation dans pering, inimas)
!===============================================================================

if (iperot.gt.0) then

  if (ivelco.eq.0) then
    call perinu                                                    &
    !==========
  ( nvar   , nscal  ,                                              &
    dt     , rtp    , rtpa   , propce , propfa , propfb ,          &
    coefa  , coefb  ,                                              &
    dudxy  )
  endif

  if (itytur.eq.3) then

    call perinr                                                   &
    !==========
 ( nvar   , nscal  ,                                              &
   dt     , rtp    , rtpa   , propce , propfa , propfb ,          &
   coefa  , coefb  ,                                              &
   drdxy  )

  endif

endif

!===============================================================================
! 3. Weakly compressible algorithm (semi-analytical):
!     Computation of scalar diffusion terms
!===============================================================================

! Diffusion terms for weakly compressible algorithm
if (idilat.eq.4.and.ipass.gt.1) then

  call diffst &
  !==========
 ( nvar   , nscal  ,                                     &
   rtp    , rtpa   , propce , propfa , propfb ,          &
   coefa  , coefb  )

endif

!===============================================================================
! 4.  ON REND LA MAIN A L'UTILISATEUR POUR LA PROGRAMMATION DES
!      GRANDEURS PHYSIQUES VARIABLES QUI LUI SONT PROPRES
!===============================================================================

ibrom = 0

if (ippmod(iphpar).ge.1) then
  call ppphyv                                                     &
  !==========
 ( nvar   , nscal  ,                                              &
   ibrom  ,                                                       &
   dt     , rtp    , rtpa   , propce , propfa , propfb ,          &
   coefa  , coefb  )

endif


! - Interface Code_Saturne
!   ======================

if (iihmpr.eq.1) then
  call uiphyv &
  !===========
( ncel, ncelet, nscaus,                                         &
  irom, iviscl, icp,    ivisls, irovar, ivivar,                 &
  isca, iscalt, iscsth, iscavr, ipproc, iviscv, itempk,         &
  p0  , t0    , ro0   , cp0   , viscl0, visls0,                 &
  rtp,    propce)
endif

call usphyv &
!==========
( nvar   , nscal  ,                                              &
  ibrom  ,                                                       &
  dt     , rtp    , rtpa   ,                                     &
  propce , propfa , propfb )

!  Density defined by a perfect gas equation of state
!  for the low-Mach algorithm
if (idilat.eq.3) then

  ! Works only with enthalpy
  if (iscalt.le.0) call csexit(1)
  if (iscsth(iscalt).ne.2) call csexit(1)
  ivarh  = isca(iscalt)
  ipcrom = ipproc(irom)

  ! Count the number of species
  nscasp = 0
  do ii = 1, nscamx
    nscasp = nscasp + iscasp(ii)
  enddo

  do iel = 1, ncel

    ! Enthalpy over Cp, with Cp specific heat variable or constant
    if (icp.gt.0) then
      ipccp  = ipproc(icp)
      xrtp =  rtp(iel,ivarh)/propce(iel,ipccp)
    else
      xrtp =  rtp(iel,ivarh)/cp0
    endif

    alpha = 0.d0

    if (nscasp.ge.2) then
      ! Deduced species
      ym = 1.d0

      do ii = 1, nscaus
        if (iscasp(ii).eq.1) then
          yk = rtp(iel, isca(ii))

          ! Clipping of the fraction yk
          yk = max(yk, 0.d0)
          yk = min(yk, 1.d0)

          alpha = alpha + yk/wmolsp(ii)

          ym = ym - yk
        endif
      enddo

      ! Clipping of remaining species
      ym = max(ym, 0.d0)

      ! Add to alpha the value due to the deduced fraction
      alpha = alpha + ym/wmolsp(0)

      ! Check if the values are correct
      if (alpha.lt.epzero .or. rair.lt.epzero .or.   &
           xrtp.lt.epzero .or. pther.lt.epzero) then
        write(nfecra,9004)
        call csexit(1)
      endif

      propce(iel,ipcrom) = pther/(alpha*rair*xrtp)

      ! Monospecies: density defined with the perfect state law
    else

      ! Check if the values are correct
      if (rair.lt.epzero .or.                       &
          xrtp.lt.epzero .or. pther.lt.epzero) then
        write(nfecra,9004)
        call csexit(1)
      endif

      propce(iel,ipcrom) = pther/(rair*xrtp)

    endif

  enddo

endif

!  ROMB SUR LES BORDS : VALEUR PAR DEFAUT (CELLE DE LA CELLULE VOISINE)

if (ibrom.eq.0) then
  ipcrom = ipproc(irom)
  ipbrom = ipprob(irom)
  do ifac = 1, nfabor
    iel = ifabor(ifac)
    propfb(ifac,ipbrom) = propce(iel ,ipcrom)
  enddo
endif

!  Au premier pas de temps du calcul
!     Si on a indique que rho (visc) etait constant
!       et qu'on l'a modifie dans usphyv, ca ne va pas
!     On se sert de irovar (ivivar) pour ecrire et lire
!       rho (visc) dans le fichier suite

if (ntcabs.eq.ntpabs+1) then

!     Masse volumique aux cellules et aux faces de bord
  iok1 = 0
  if (irovar.eq.0) then
    ipcrom = ipproc(irom)
    ipbrom = ipprob(irom)
    do iel = 1, ncel
      if ( abs(propce(iel ,ipcrom)-ro0   ).gt.epzero) then
        iok1 = 1
      endif
    enddo
    do ifac = 1, nfabor
      if ( abs(propfb(ifac,ipbrom)-ro0   ).gt.epzero) then
        iok1 = 1
      endif
    enddo
  endif
  if (iok1.ne.0) then
    write(nfecra,9001)
  endif

!     Viscosite moleculaire aux cellules
  iok2 = 0
  if (ivivar.eq.0) then
    ipcvis = ipproc(iviscl)
    do iel = 1, ncel
      if ( abs(propce(iel ,ipcvis)-viscl0).gt.epzero) then
        iok2 = 1
      endif
    enddo
  endif
  if (iok2.ne.0) then
    if ( ippmod(icompf) .ge. 0 ) then
      write(nfecra,9003)
    else
      write(nfecra,9002)
    endif
  endif

  if (iok1.ne.0.or.iok2.ne.0) then
    call csexit(1)
  endif

endif

!===============================================================================
! 3.  CALCUL DE LA VISCOSITE TURBULENTE
!===============================================================================

if     (iturb.eq. 0) then

! 3.1 LAMINAIRE
! ==============

  ipcvst = ipproc(ivisct)

  do iel = 1, ncel
    propce(iel,ipcvst) = 0.d0
  enddo

elseif (iturb.eq.10) then

! 3.2 LONGUEUR DE MELANGE
! ========================

  call vislmg &
  !==========
 ( nvar   , nscal  ,                                              &
   ncepdc , ncetsm ,                                              &
   icepdc , icetsm , itypsm ,                                     &
   dt     , rtp    , rtpa   , propce , propfa , propfb ,          &
   coefa  , coefb  , ckupdc , smacel )

elseif (itytur.eq.2) then

! 3.3 K-EPSILON
! ==============

  ipcvst = ipproc(ivisct)
  ipcrom = ipproc(irom  )

  do iel = 1, ncel
    xk = rtp(iel,ik)
    xe = rtp(iel,iep)
    propce(iel,ipcvst) = propce(iel,ipcrom)*cmu*xk**2/xe
  enddo

elseif (itytur.eq.3) then

! 3.4 Rij-EPSILON
! ================

  ipcvst = ipproc(ivisct)
  ipcrom = ipproc(irom  )

  do iel = 1, ncel
    xk = 0.5d0*(rtp(iel,ir11)+rtp(iel,ir22)+rtp(iel,ir33))
    xe = rtp(iel,iep)
    propce(iel,ipcvst) = propce(iel,ipcrom)*cmu*xk**2/xe
  enddo

elseif (iturb.eq.40) then

! 3.5 LES Smagorinsky
! ===================


  call vissma &
  !==========
 ( nvar   , nscal  ,                                              &
   ncepdc , ncetsm ,                                              &
   icepdc , icetsm , itypsm ,                                     &
   dt     , rtp    , rtpa   , propce , propfa , propfb ,          &
   coefa  , coefb  , ckupdc , smacel )

elseif (iturb.eq.41) then

! 3.6 LES dynamique
! =================


  call visdyn &
  !==========
 ( nvar   , nscal  ,                                              &
   ncepdc , ncetsm ,                                              &
   icepdc , icetsm , itypsm ,                                     &
   dt     , rtp    , rtpa   , propce , propfa , propfb ,          &
   coefa  , coefb  , ckupdc , smacel ,                            &
   propce(1,ipproc(ismago)) )

elseif (iturb.eq.42) then

! 3.7 LES WALE
! ============


  call viswal &
  !==========
 ( nvar   , nscal  ,                                              &
   ncepdc , ncetsm ,                                              &
   icepdc , icetsm , itypsm ,                                     &
   dt     , rtp    , rtpa   , propce , propfa , propfb ,          &
   coefa  , coefb  , ckupdc , smacel )

elseif (itytur.eq.5) then

! 3.8 v2f (phi-model and BL-v2/k)
! ===============================

  if (iturb.eq.50) then

    ipcvis = ipproc(iviscl)
    ipcvst = ipproc(ivisct)
    ipcrom = ipproc(irom  )

    do iel = 1, ncel
      xk = rtp(iel,ik)
      xe = rtp(iel,iep)
      xrom = propce(iel,ipcrom)
      xnu = propce(iel,ipcvis)/xrom
      ttke = xk / xe
      ttmin = cv2fct*sqrt(xnu/xe)
      tt = max(ttke,ttmin)
      propce(iel,ipcvst) = cv2fmu*xrom*tt*rtp(iel,iphi)*rtp(iel,ik)
    enddo

  else if (iturb.eq.51) then

    call visv2f &
    !==========
   ( nvar   , nscal  ,                                              &
     ncepdc , ncetsm ,                                              &
     icepdc , icetsm , itypsm ,                                     &
     dt     , rtp    , rtpa   , propce , propfa , propfb ,          &
     coefa  , coefb  , ckupdc , smacel )

  endif

elseif (iturb.eq.60) then

! 3.9 K-OMEGA SST
! ===============

  call vissst &
  !==========
 ( nvar   , nscal  ,                                              &
   ncepdc , ncetsm ,                                              &
   icepdc , icetsm , itypsm ,                                     &
   dt     , rtp    , rtpa   , propce , propfa , propfb ,          &
   coefa  , coefb  , ckupdc , smacel )

elseif (iturb.eq.70) then

! 3.10 SPALART -ALLMARAS
! ======================

  cv13 = csav1**3

  ipcvst = ipproc(ivisct)
  ipcrom = ipproc(irom  )
  ipcvis = ipproc(iviscl)

  do iel = 1, ncel
    xrom = propce(iel,ipcrom)
    nusa = rtp(iel,inusa)
    xi3  = (xrom*nusa/propce(iel,ipcvis))**3
    fv1  = xi3/(xi3+cv13)
    propce(iel,ipcvst) = xrom*nusa*fv1
  enddo

endif

!===============================================================================
! 4. Symmetric tensor diffusivity
!===============================================================================
iok = 0
do ivar = 1, nvar
  if (idften(ivar).eq.6) iok = 1
enddo

if (iok.eq.1) then
  if (itytur.eq.3) then

    ipcrom = ipproc(irom)

    do iel = 1, ncel
      !FIXME it should be csrij/sigmae instead of crijep
      trrij = 0.5d0*(rtp(iel,ir11)+rtp(iel,ir22)+rtp(iel,ir33))
      csteps  = propce(iel,ipcrom) * csrij * trrij / rtp(iel,iep)

      visten(1,iel) = csteps*rtp(iel,ir11)
      visten(2,iel) = csteps*rtp(iel,ir22)
      visten(3,iel) = csteps*rtp(iel,ir33)
      visten(4,iel) = csteps*rtp(iel,ir12)
      visten(5,iel) = csteps*rtp(iel,ir13)
      visten(6,iel) = csteps*rtp(iel,ir23)
    enddo

  else

    do iel = 1, ncel
      visten(1,iel) = 0.d0
      visten(2,iel) = 0.d0
      visten(3,iel) = 0.d0
      visten(4,iel) = 0.d0
      visten(5,iel) = 0.d0
      visten(6,iel) = 0.d0
    enddo

  endif
endif

!===============================================================================
! 5. User modification of the turbulent viscosity and symmetric tensor
!    diffusivity
!===============================================================================

call usvist &
!==========
( nvar   , nscal  ,                                              &
  ncepdc , ncetsm ,                                              &
  icepdc , icetsm , itypsm ,                                     &
  dt     , rtp    , rtpa   , propce , propfa , propfb ,          &
  ckupdc , smacel )

!===============================================================================
! 6. Clipping of the turbulent viscosity in dynamic LES
!===============================================================================

! Pour la LES en modele dynamique on clippe la viscosite turbulente de maniere
! a ce que mu+mu_t soit positif, .e. on autorise mu_t legerement negatif
! La diffusivite turbulente des scalaires (mu_t/sigma), elle, sera clippee a 0
! dans covofi

if (iturb.eq.41) then
  ipcvis = ipproc(iviscl)
  ipcvst = ipproc(ivisct)
  iclipc = 0
  do iel = 1, ncel
    viscto = propce(iel,ipcvis) + propce(iel,ipcvst)
    if (viscto.lt.0.d0) then
      propce(iel,ipcvst) = 0.d0
      iclipc = iclipc + 1
    endif
  enddo
  if (iwarni(iu).ge.1) then
    if (irangp.ge.0) then
      call parcpt(iclipc)
      !==========
    endif
    write(nfecra,1000) iclipc
  endif
endif

!===============================================================================
! 6. User modification of the mesh viscosity in ALE
!===============================================================================

if (iale.eq.1.and.ntcabs.eq.0) then

  ! - Interface Code_Saturne
  !   ======================

  if (iihmpr.eq.1) then

    call uivima                       &
    !==========
  ( ncel,                             &
    propce(1,ipproc(ivisma(1))),      &
    propce(1,ipproc(ivisma(2))),      &
    propce(1,ipproc(ivisma(3))),      &
    xyzcen, dtref, ttcabs, ntcabs )

  endif

  call usvima                                                     &
  !==========
 ( nvar   , nscal  ,                                              &
   dt     , rtp    , rtpa   , propce , propfa , propfb ,          &
   propce(1,ipproc(ivisma(1))) ,                                  &
   propce(1,ipproc(ivisma(2))) , propce(1,ipproc(ivisma(3))) )

endif

!===============================================================================
! 7.  IMPRESSIONS DE CONTROLE DES VALEURS ENTREES PAR L'UTILISATEUR
!===============================================================================

! ---> Calcul des bornes des variables et impressions

! Indicateur d'erreur
iok = 0

! Rang des variables dans PROPCE
ipcrom = ipproc(irom)
ipcvis = ipproc(iviscl)
ipcvst = ipproc(ivisct)
if (icp.gt.0) then
  ipccp  = ipproc(icp   )
  nn     = 4
else
  ipccp = 0
  nn    = 3
endif

! Rang des variables dans PROPFB
ipbrom = ipprob(irom)

! Min et max sur les cellules
do ii = 1, nn
  ivar = 0
  if (ii.eq.1) ivar = ipcrom
  if (ii.eq.2) ivar = ipcvis
  if (ii.eq.3) ivar = ipcvst
  if (ii.eq.4) ivar = ipccp
  if (ivar.gt.0) then
    varmx(ii) = propce(1,ivar)
    varmn(ii) = propce(1,ivar)
    do iel = 2, ncel
      varmx(ii) = max(varmx(ii),propce(iel,ivar))
      varmn(ii) = min(varmn(ii),propce(iel,ivar))
    enddo
    if (irangp.ge.0) then
      call parmax (varmx(ii))
      !==========
      call parmin (varmn(ii))
      !==========
    endif
  endif
enddo

! Min et max sur les faces de bord (masse volumique uniquement)
ii   = 1
ivar = ipbrom
do ifac = 1, nfabor
  varmx(ii) = max(varmx(ii),propfb(ifac,ivar))
  varmn(ii) = min(varmn(ii),propfb(ifac,ivar))
enddo
if (irangp.ge.0) then
  call parmax (varmx(ii))
  !==========
  call parmin (varmn(ii))
  !==========
endif

! Impressions
iok1 = 0
do ii = 1, nn
  if (ii.eq.1) chaine = nomvar(ipppro(ipproc(irom  )))
  if (ii.eq.2) chaine = nomvar(ipppro(ipproc(iviscl)))
  if (ii.eq.3) chaine = nomvar(ipppro(ipproc(ivisct)))
  if (ii.eq.4) chaine = nomvar(ipppro(ipproc(icp   )))
  if (iwarni(iu).ge.1.or.ipass.eq.1.or.varmn(ii).lt.0.d0) then
    if (iok1.eq.0) then
      write(nfecra,3010)
      iok1 = 1
    endif
    if ((ii.ne.3).or.(iturb.ne.0))                          &
         write(nfecra,3011)chaine(1:16),varmn(ii),varmx(ii)
  endif
enddo
if (iok1.eq.1) write(nfecra,3012)

! Verifications de valeur physique

! Masse volumique definie
ii = 1
chaine = nomvar(ipppro(ipproc(irom  )))
if (varmn(ii).lt.0.d0) then
  write(nfecra,9011)chaine(1:16),varmn(ii)
  iok = iok + 1
endif

! Viscosite moleculaire definie
ii = 2
chaine = nomvar(ipppro(ipproc(iviscl)))
if (varmn(ii).lt.0.d0) then
  write(nfecra,9011)chaine(1:16),varmn(ii)
  iok = iok + 1
endif

! Viscosite turbulente definie
! on ne clippe pas mu_t en modele LES dynamique, car on a fait
! un clipping sur la viscosite totale
ii = 3
chaine = nomvar(ipppro(ipproc(ivisct)))
if (varmn(ii).lt.0.d0.and.iturb.ne.41) then
  write(nfecra,9012)varmn(ii)
  iok = iok + 1
endif

! Chaleur specifique definie
if (icp.gt.0) then
  ii = 4
  chaine = nomvar(ipppro(ipproc(icp   )))
  if (varmn(ii).lt.0.d0) then
    iisct = 0
    do iscal = 1, nscal
      if (iscsth(iscal).ne.0) then
        iisct = 1
      endif
    enddo
    if (iisct.eq.1) then
      write(nfecra,9011)chaine(1:16),varmn(ii)
      iok = iok + 1
    endif
  endif
endif

! ---> Calcul des bornes des scalaires et impressions

if (nscal.ge.1) then

  iok1 = 0
  do iscal = 1, nscal

    if (ivisls(iscal).gt.0) then
      ipcvis = ipproc(ivisls(iscal))
    else
      ipcvis = 0
    endif

    vismax(iscal) = -grand
    vismin(iscal) =  grand
    if (ipcvis.gt.0) then
      do iel = 1, ncel
        vismax(iscal) = max(vismax(iscal),propce(iel,ipcvis))
        vismin(iscal) = min(vismin(iscal),propce(iel,ipcvis))
      enddo
      if (irangp.ge.0) then
        call parmax (vismax(iscal))
        !==========
        call parmin (vismin(iscal))
        !==========
      endif
    else
      vismax(iscal) = visls0(iscal)
      vismin(iscal) = visls0(iscal)
    endif

    ivar = isca(iscal)
    if (iwarni(ivar).ge.1.or.ipass.eq.1.or.vismin(iscal).le.0.d0) then
      chaine = nomvar(ipprtp(ivar))
      if (iok1.eq.0) then
        write(nfecra,3110)
        iok1 = 1
      endif
      write(nfecra,3111) chaine(1:16),iscal,vismin(iscal),vismax(iscal)
    endif

  enddo
  if (iok1.eq.1) write(nfecra,3112)

! Verifications de valeur physique

! IOK a deja ete initialise

  do iscal = 1, nscal

    ivar = isca(iscal)

    if (vismin(iscal).lt.0.d0) then
      chaine = nomvar(ipprtp(ivar))
      write(nfecra,9111)chaine(1:16),iscal,vismin(iscal)
      iok = iok + 1
    endif

  enddo

endif

! ---> Calcul des bornes de viscosite de maillage en ALE

if (iale.eq.1.and.ntcabs.eq.0) then

  iok1 = 0
  nn = 1
  if (iortvm.eq.1) nn = 3
  do ii = 1, nn
    ipcvma = ipproc(ivisma(ii))

! Min et max sur les cellules
    varmx(1) = propce(1,ipcvma)
    varmn(1) = propce(1,ipcvma)
    do iel = 2, ncel
      varmx(1) = max(varmx(1),propce(iel,ipcvma))
      varmn(1) = min(varmn(1),propce(iel,ipcvma))
    enddo
    if (irangp.ge.0) then
      call parmax (varmx(1))
      !==========
      call parmin (varmn(1))
      !==========
    endif

! Impressions
    chaine = nomvar(ipppro(ipcvma))
    if (iwarni(iuma).ge.1.or.ipass.eq.1.or.varmn(1).lt.0.d0) then
      if (iok1.eq.0) then
        write(nfecra,3210)
        iok1 = 1
      endif
      write(nfecra,3211)chaine(1:16),varmn(1),varmx(1)
    endif

! Verifications de valeur physique

! Viscosite de maillage definie
    chaine = nomvar(ipppro(ipcvma))
    if (varmn(1).le.0.d0) then
      write(nfecra,9211) varmn(1)
      iok = iok + 1
    endif

  enddo

  if (iok1.eq.1) write(nfecra,3212)

endif

! --->  arret eventuel

if (iok.ne.0) then
  write(nfecra,9999)iok
  call csexit (1)
endif

!===============================================================================
! 8.  ECHANGES
!===============================================================================

! Pour navsto et vissec on a besoin de ROM dans le halo


ipcrom = ipproc(irom)

if (irangp.ge.0.or.iperio.eq.1) then
  call synsca(propce(1,ipcrom))
  !==========
endif

!----
! FORMATS
!----

#if defined(_CS_LANG_FR)

 1000 format(                                                     &
' Nb de clippings de la viscosite totale (mu+mu_t>0) :',i10,/)
 3010 format(                                                     &
' -----------------------------------------',                   /,&
' Propriete          Valeur min  Valeur max',                   /,&
' -----------------------------------------                   '  )
 3011 format(                                                     &
 2x,    a16,      e12.4,      e12.4                              )
 3012 format(                                                     &
' -----------------------------------------',                   /)
 3110 format(                                                     &
' --- Diffusivite :',                                           /,&
' -----------------------------------------------',             /,&
' Scalaire         Numero  Valeur min  Valeur max',             /,&
' -----------------------------------------------'               )
 3111 format(                                                     &
 1x,    a16,   i7,      e12.4,      e12.4                        )
 3112 format(                                                     &
' ---------------------------------------',                     /)
 3210 format(                                                     &
' --- Viscosite de maillage (methode ALE)',                     /,&
' -----------------------------------------',                   /,&
' Propriete          Valeur min  Valeur max',                   /,&
' -----------------------------------------'                     )
 3211 format(                                                     &
 2x,    a16,      e12.4,      e12.4                              )
 3212 format(                                                     &
' -----------------------------------------',                   /)

 9001  format(                                                    &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION : ARRET LORS DU CALCUL DES GRANDEURS PHYSIQUES',/,&
'@    =========',                                               /,&
'@    INCOHERENCE ENTRE LES PARAMETRES ET LA MASSE VOLUMIQUE',  /,&
'@',                                                            /,&
'@  On a indique que la masse volumique etait',                 /,&
'@     constante (IROVAR=0) mais on a modifie ses',             /,&
'@     valeurs aux cellules ou aux faces de bord.',             /,&
'@',                                                            /,&
'@  Le calcul ne sera pas execute.',                            /,&
'@',                                                            /,&
'@  Verifier l''interface, cs_user_parameters.f90, et usphyv.', /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 9002  format(                                                    &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION : ARRET LORS DU CALCUL DES GRANDEURS PHYSIQUES',/,&
'@    =========',                                               /,&
'@    INCOHERENCE ENTRE LES PARAMETRES ET',                     /,&
'@                                    LA VISCOSITE MOLECULAIRE',/,&
'@',                                                            /,&
'@  On a indique que la viscosite moleculaire',                 /,&
'@     etait constante (IVIVAR=0) mais on a modifie ses',/,&
'@     valeurs aux cellules.',                                  /,&
'@',                                                            /,&
'@  Le calcul ne sera pas execute.',                            /,&
'@',                                                            /,&
'@  Verifier l''interface, cs_user_parameters.f90, et usphyv.', /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 9003  format(                                                    &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION : ARRET LORS DU CALCUL DES GRANDEURS PHYSIQUES',/,&
'@    =========',                                               /,&
'@    MODULE COMPRESSIBLE',                                     /,&
'@    INCOHERENCE ENTRE USCFPV ET USCFX1 POUR',                 /,&
'@                                    LA VISCOSITE MOLECULAIRE',/,&
'@',                                                            /,&
'@  En compressible la viscosite moleculaire est constante par',/,&
'@     defaut (IVIVAR=0) et la valeur de IVIVAR n''a',   /,&
'@     pas ete modifiee dans uscfx1. Pourtant, on a modifie',   /,&
'@     les valeurs de la viscosite moleculaire dans uscfpv.',   /,&
'@',                                                            /,&
'@  Le calcul ne sera pas execute.',                            /,&
'@',                                                            /,&
'@  Verifier uscfx1 et uscfpv.',                                /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
9004  format( &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION : ARRET LORS DU CALCUL DES GRANDEURS PHYSIQUES',/,&
'@    =========',                                               /,&
'@    OPTION IDILAT 3',                                         /,&
'@    LE CALCUL ENGENDRE UN RHO NEGATIF',                       /,&
'@',                                                            /,&
'@  Le calcul ne sera pas execute.',                            /,&
'@',                                                            /,&
'@  Verifier uscfx1 et uscfpv.',                                /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 9011  format(                                                    &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION : ARRET LORS DU CALCUL DES GRANDEURS PHYSIQUES',/,&
'@    =========',                                               /,&
'@    LA PROPRIETE PHYSIQUE', a16  ,' N A PAS ETE',             /,&
'@                                       CORRECTEMENT DEFINIE.',/,&
'@',                                                            /,&
'@  Le calcul ne sera pas execute.',                            /,&
'@',                                                            /,&
'@  La propriete physique identifiee ci-dessus est variable et',/,&
'@    le minimum atteint est', e12.4                           ,/,&
'@  Verifier que cette propriete a ete definie et',             /,&
'@    que la loi adoptee conduit a des valeurs correctes.',     /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 9012  format(                                                    &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION : ARRET LORS DU CALCUL DES GRANDEURS PHYSIQUES',/,&
'@    =========',                                               /,&
'@    LA VISCOSITE TURBULENTE',                                 /,&
'@                           N A PAS ETE CORRECTEMENT DEFINIE.',/,&
'@',                                                            /,&
'@  Le calcul ne sera pas execute.',                            /,&
'@',                                                            /,&
'@  Le minimum atteint est', e12.4                             ,/,&
'@  Verifier le cas echeant la definition de la masse',         /,&
'@    volumique et la modification de la viscosite turbulente', /,&
'@    dans usvist.',                                            /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)

 9111  format(                                                    &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION : ARRET LORS DU CALCUL DES GRANDEURS PHYSIQUES',/,&
'@    =========',                                               /,&
'@    LA DIFFUSIVITE DU SCALAIRE', a16                         ,/,&
'@       (SCALAIRE NUMERO', i10   ,') N A PAS ETE',             /,&
'@                                       CORRECTEMENT DEFINIE.',/,&
'@',                                                            /,&
'@  Le calcul ne sera pas execute.',                            /,&
'@',                                                            /,&
'@  La propriete physique identifiee ci-dessus est variable et',/,&
'@    le minimum atteint est', e12.4                           ,/,&
'@  Verifier que cette propriete a bien ete definie et',        /,&
'@    que la loi adoptee conduit a des valeurs correctes.',     /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 9211  format(                                                    &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION : ARRET LORS DU CALCUL DES GRANDEURS PHYSIQUES',/,&
'@    =========',                                               /,&
'@    LA VISCOSITE DE MAILLAGE N A PAS ETE',                    /,&
'@                                       CORRECTEMENT DEFINIE.',/,&
'@',                                                            /,&
'@  Le calcul ne sera pas execute.',                            /,&
'@',                                                            /,&
'@  Le minimum atteint est', e12.4                             ,/,&
'@  Verifier le cas echeant la modification de la viscosite',   /,&
'@    dans usvima ou dans l''interface graphique.',             /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 9999 format(                                                     &
'@',                                                            /,&
'@',                                                            /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ ATTENTION : ARRET LORS DU CALCUL DES GRANDEURS PHYSIQUES',/,&
'@    =========',                                               /,&
'@    DES GRANDEURS PHYSIQUES ONT DES VALEURS INCORRECTES',     /,&
'@',                                                            /,&
'@  Le calcul ne sera pas execute (',i10,' erreurs).',          /,&
'@',                                                            /,&
'@  Se reporter aux impressions precedentes pour plus de',      /,&
'@    renseignements.',                                         /,&
'@  Verifier les definitions et lois definies',                 /,&
'@    dans usphyv ou dans l''interface graphique.',             /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)

#else

 1000 format(                                                     &
' Nb of clippings for the effective viscosity (mu+mu_t>0):',i10,/)
 3010 format(                                                     &
' -----------------------------------------',                   /,&
' Property           Min. value  Max. value',                   /,&
' -----------------------------------------'                     )
 3011 format(                                                     &
 2x,    a16,      e12.4,      e12.4                              )
 3012 format(                                                     &
' -----------------------------------------',                   /)
 3110 format(                                                     &
' --- Diffusivity:',                                            /,&
' -----------------------------------------------',             /,&
' Scalar           Number  Min. value  Max. value',             /,&
' -----------------------------------------------'               )
 3111 format(                                                     &
 1x,    a16,   i7,      e12.4,      e12.4                        )
 3112 format(                                                     &
' -----------------------------------------------',             /)
 3210 format(                                                     &
' --- Mesh viscosity (ALE method)',                             /,&
' -----------------------------------------',                   /,&
' Property           Min. value  Max. value',                   /,&
' -----------------------------------------'                     )
 3211 format(                                                     &
 2x,    a16,      e12.4,      e12.4                              )
 3212 format(                                                     &
' -----------------------------------------',                   /)

 9001  format(                                                    &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING: ABORT IN THE PHYSICAL QUANTITIES COMPUTATION',   /,&
'@    ========',                                                /,&
'@    INCOHERENCY BETWEEN PARAMETERS FOR THE DENSITY.',         /,&
'@',                                                            /,&
'@  The density has been declared constant',                    /,&
'@     (IROVAR=0) but its value has been modified',             /,&
'@     in cells or at boundary faces.',                         /,&
'@',                                                            /,&
'@  The calculation will not be run.',                          /,&
'@',                                                            /,&
'@  Check the interface, cs_user_parameters.f90, and usphyv',   /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 9002  format(                                                    &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING: ABORT IN THE PHYSICAL QUANTITIES COMPUTATION',   /,&
'@    ========',                                                /,&
'@    INCOHERENCY BETWEEN PARAMETERS FOR',                      /,&
'@                                     THE MOLECULAR VISCOSITY',/,&
'@',                                                            /,&
'@  The molecular viscosity has been declared constant',        /,&
'@     (IVIVAR=0) but its value has been  modified in cells',   /,&
'@     or at boundary faces.',                                  /,&
'@',                                                            /,&
'@  The calculation will not be run.',                          /,&
'@',                                                            /,&
'@  Check the interface, cs_user_parameters.f90, and usphyv',   /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 9003  format(                                                    &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING: ABORT IN THE PHYSICAL QUANTITIES COMPUTATION',   /,&
'@    ========',                                                /,&
'@    INCOHERENCY BETWEEN USCFPV AND USCFX1 FOR',               /,&
'@                                     THE MOLECULAR VISCOSITY',/,&
'@',                                                            /,&
'@  In the compressible module, the molecular viscosity is',    /,&
'@     constant by default (IVIVAR=0) and the value',    /,&
'@     of IVIVAR  has not been modified in uscfx1. Yet, its',   /,&
'@     value has been modified in uscfpv.',                     /,&
'@',                                                            /,&
'@  The calculation will not be run.',                          /,&
'@',                                                            /,&
'@  Verify uscfx1 and uscfpv.',                                 /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 9004  format( &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING: ABORT IN THE PHYSICAL QUANTITIES COMPUTATION',   /,&
'@    ========',                                                /,&
'@    OPTION IDILAT 3',                                         /,&
'@    THE COMPUTATION MAKE RHO NEGATIVE',                       /,&
'@',                                                            /,&
'@  The calculation will not be run.',                          /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 9011  format(                                                    &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING: ABORT IN THE PHYSICAL QUANTITIES COMPUTATION',   /,&
'@    ========',                                                /,&
'@    THE PHYSICAL PROPERTY', a16  ,' HAS NOT BEEN',            /,&
'@                                          CORRECTLY DEFINED.',/,&
'@',                                                            /,&
'@  The calculation will not be run.',                          /,&
'@',                                                            /,&
'@  The physical property identified is variable and the',      /,&
'@    minimum reached is', e12.4                               ,/,&
'@  Verify that this property has been defined and',            /,&
'@    that the chosen law leads to correct values.',            /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 9012  format(                                                    &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING: ABORT IN THE PHYSICAL QUANTITIES COMPUTATION',   /,&
'@    ========',                                                /,&
'@    THE TURBULENT VISCOSITY HAS NOT BEEN CORRECTLY DEFINED.', /,&
'@',                                                            /,&
'@  The calculation will not be run.',                          /,&
'@',                                                            /,&
'@  The  minimum reached is', e12.4                            ,/,&
'@  Verify the density definition  and the turbulent viscosity',/,&
'@    modification in usvist (if any).',                        /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)

 9111  format(                                                    &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING: ABORT IN THE PHYSICAL QUANTITIES COMPUTATION',   /,&
'@    ========',                                                /,&
'@    THE DIFFUSIVITY OF THE SCALAR', a16                      ,/,&
'@       (SCALAR NUMBER', i10   ,') HAS NOT BEEN',              /,&
'@                                          CORRECTLY DEFINED.',/,&
'@',                                                            /,&
'@  The calculation will not be run.',                          /,&
'@',                                                            /,&
'@  The physical property identified is variable and the',      /,&
'@    minimum reached is', e12.4                               ,/,&
'@  Verify that this property has been defined in usphyv and',  /,&
'@    that the chosen law leads to correct values.',            /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 9211  format(                                                    &
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING: ABORT IN THE PHYSICAL QUANTITIES COMPUTATION',   /,&
'@    ========',                                                /,&
'@    THE MESH VISCOSITY HAS NOT BEEN CORRECTLY DEFINED.',      /,&
'@',                                                            /,&
'@  The calculation will not be run.',                          /,&
'@',                                                            /,&
'@  The  minimum reached is', e12.4                            ,/,&
'@  Verify that this property has been defined in usvima and',  /,&
'@    that the chosen law leads to correct values.',            /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)
 9999 format(                                                     &
'@',                                                            /,&
'@',                                                            /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /,&
'@ @@ WARNING: ABORT IN THE PHYSICAL QUANTITIES COMPUTATION',   /,&
'@    ========',                                                /,&
'@    SOME PHYSICAL QUANTITIES HAVE INCORRECT VALUES',          /,&
'@',                                                            /,&
'@  The calculation will not be run (',i10,' errors).',         /,&
'@',                                                            /,&
'@  Refer to previous warnings for further information.',       /,&
'@  Verify usphyv.',                                            /,&
'@',                                                            /,&
'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',/,&
'@',                                                            /)

#endif

!----
! End
!----

return
end subroutine
