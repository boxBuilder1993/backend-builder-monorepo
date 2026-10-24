# Postgres backups: every app database, dumped nightly, kept locally with retention, uploaded to a
# private store when ~/.config/backend-builder/backup.env names one. See docs/backups.md.
.PHONY: pg-backup pg-backups pg-fetch pg-restore pg-backup-install pg-backup-remove

pg-backup:       ## dump every app database (or DB=x), prune, upload when BACKUP_REPO is set
	@bin/bb-backup dump $(DB)

pg-backups:      ## list local and remote dumps (DB=x for one)
	@bin/bb-backup list $(DB)

pg-fetch:        ## pull the newest remote dump (or NAME=file) for DB=x into the local directory
	@test -n "$(DB)" || { echo "usage: make pg-fetch DB=<database> [NAME=<dump>]"; exit 2; }
	@bin/bb-backup fetch $(DB) $(NAME)

pg-restore:      ## restore DB=x from FILE=<dump> into INTO=<db> (default x_restored; INTO=x replaces the live db, asks first)
	@test -n "$(DB)" && test -n "$(FILE)" || { echo "usage: make pg-restore DB=<database> FILE=<dump> [INTO=<db>]"; exit 2; }
	@bin/bb-backup restore $(DB) $(FILE) $(INTO)

pg-backup-install:  ## run pg-backup daily at 03:00 (systemd user timer)
	@mkdir -p $(UNIT_DIR)
	@sed 's#@REPO@#$(REPO)#g' systemd/bb-backup.service > $(UNIT_DIR)/bb-backup.service
	@cp systemd/bb-backup.timer $(UNIT_DIR)/bb-backup.timer
	@systemctl --user daemon-reload && systemctl --user enable --now bb-backup.timer
	@systemctl --user list-timers bb-backup.timer --no-pager | head -2

pg-backup-remove:   ## stop the daily backup timer
	-@systemctl --user disable --now bb-backup.timer 2>/dev/null
